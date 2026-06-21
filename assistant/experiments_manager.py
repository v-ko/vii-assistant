from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image
from PySide6.QtGui import QImage

from assistant.constants import EXPERIMENTS_DIR
from assistant.experiments.actions import (
    clear_experiment_context,
    mark_experiment_finished,
    set_experiment_mode,
    submit_experiment_context,
    update_experiment_overlay,
)
from assistant.experiments.data_loaders import (
    DATA_LOADER_REGISTRY,
    DataLoader,
    I2EBenchDataLoader,
    ScreenGrabDataLoader,
    ScreenSpotProDataLoader,
    SyntheticRectangleDataLoader,
)
from assistant.experiments.evaluation import (
    IOU_CORRECT_THRESHOLD,
    center_in_box,
    compute_iou,
    parse_bbox_from_response,
)
from assistant.experiments.utils import (
    append_result_jsonl,
    save_results_json,
    save_sample_image,
)
from assistant.facade import vii
from assistant.inference.context import TextMessage
from assistant.model.experiment_config import ExperimentConfig
from assistant.services.segment_parsing import hfi
from assistant.util import Shape
from assistant.view_states.overlay import DisplayTransform

if TYPE_CHECKING:
    from assistant.services.hybrid_segment_service import ChainResult

log = logging.getLogger(__name__)

INFERENCE_TIMEOUT_S = 40  # per-step inference timeout
RECONNECT_DELAY_S = 5.0
MAX_RECONNECT_ATTEMPTS = 30
MAX_CONSECUTIVE_FAILURES = 5


class ExperimentState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    FINISHED = "finished"


class ExperimentsManager:
    def __init__(self, app_view_model=None):
        self._app_view_model = app_view_model
        self._state = ExperimentState.IDLE
        self._current_step = 0
        self._waiting_for_response = False
        self._results: list[dict] = []
        self._config: ExperimentConfig | None = None
        self._agent: str = ""
        self._data_loader: DataLoader | None = None
        self.config_path: Path | None = None

        # Run-all state
        self._run_all_active = False
        self._run_all_task: asyncio.Task | None = None
        self._cancelled = False
        self._output_dir: Path | None = None
        self._current_sample_image: Image.Image | None = None
        self._current_sample_gt_bbox: tuple[int, int, int, int] | None = None

        # Future owned by HybridSegmentService.run_chain; resolved when the
        # assistant chain settles (TURN_COMPLETED / MAX_TURNS) or fails.
        self._step_future: asyncio.Future | None = None

        # React to WS disconnects by rejecting the step future immediately
        vii.app.view_state.settings_VS.session_state_changed.connect(
            self._on_session_state_changed
        )

    def start(self, config: ExperimentConfig) -> None:
        """Start a new experiment run with the given config."""
        if self._state == ExperimentState.RUNNING:
            raise RuntimeError("Experiment already running; stop it first")
        self._config = config
        self._current_step = 0
        self._results = []
        self._agent = vii.active_agent or ""
        self._data_loader = self._create_data_loader(config)
        self._state = ExperimentState.RUNNING
        self._cancelled = False
        set_experiment_mode()
        # Apply the per-agent turn cap (instance override; falls back to the
        # service default when the config has no entry for the active agent).
        hss = vii.project_manager.hybrid_segment_service
        agent_max_turns = config.max_turns.get(self._agent)
        if agent_max_turns is not None:
            hss.MAX_TURNS = agent_max_turns
        log.info(f"Started experiment: {config.name}")

    def cancel(self) -> None:
        """Cancel a run-all loop. Current step finishes then loop exits."""
        self._cancelled = True
        log.info("Experiment run cancelled")

    @property
    def progress(self) -> tuple[int, int]:
        """Returns (current_step, total_samples) respecting config range."""
        if not self._data_loader:
            return (0, 0)
        total = len(self._data_loader)
        end = (
            self._config.end_index
            if self._config and self._config.end_index is not None
            else total
        )
        end = min(end, total)
        start = (
            self._config.start_index
            if self._config and self._config.start_index is not None
            else 0
        )
        return (self._current_step - start, end - start)

    @property
    def is_run_all_active(self) -> bool:
        return self._run_all_active

    @property
    def state(self) -> ExperimentState:
        return self._state

    @property
    def running(self) -> bool:
        return self._state in (ExperimentState.RUNNING, ExperimentState.FINISHED)

    def _abort_step(self, reason: str):
        """Reject the pending step future and reset waiting state."""
        log.error(f"Experiment step aborted: {reason}")
        self._waiting_for_response = False
        # Reject the pending future so the await raises
        if self._step_future and not self._step_future.done():
            self._step_future.set_exception(RuntimeError(reason))
        vii.app.view_state.overlay_VS.clear()

    def _on_session_state_changed(self, state: str) -> None:
        """Reject step future on WS disconnect so run_all loop can reconnect."""
        if (
            state == "disconnected"
            and self._step_future
            and not self._step_future.done()
        ):
            self._waiting_for_response = False
            vii.project_manager.hybrid_segment_service.fail_chain_disconnected()

    def _create_data_loader(self, config: ExperimentConfig) -> DataLoader:
        """Instantiate the data loader specified in the experiment config."""
        loader_name = config.data_loader
        if not loader_name:
            raise ValueError("Experiment config must specify 'data_loader'")
        loader_cls = DATA_LOADER_REGISTRY.get(loader_name)
        if loader_cls is None:
            raise ValueError(
                f"Unknown data_loader '{loader_name}'. "
                f"Available: {list(DATA_LOADER_REGISTRY)}"
            )

        if loader_cls is ScreenGrabDataLoader:
            return ScreenGrabDataLoader(
                prompt=config.prompt,
            )

        if loader_cls is ScreenSpotProDataLoader:
            if not config.dataset_path:
                raise ValueError("ScreenSpotPro requires 'dataset_path' in config")
            return ScreenSpotProDataLoader(
                dataset_path=config.dataset_path,
                prompt_template=config.prompt_template
                or "Locate the element: {instruction}",
            )

        if loader_cls is I2EBenchDataLoader:
            if not config.dataset_path:
                raise ValueError("I2EBench requires 'dataset_path' in config")
            return I2EBenchDataLoader(
                dataset_path=config.dataset_path,
                prompt_template=config.prompt_template
                or "Locate the element: {instruction}",
            )

        if loader_cls is SyntheticRectangleDataLoader:
            gen_params = config.generation_params or {}
            return SyntheticRectangleDataLoader(
                num_samples=gen_params.get("num_samples", 50),
                min_size=gen_params.get("min_size", 0.05),
                max_size=gen_params.get("max_size", 0.4),
                image_width=gen_params.get("image_width", 1920),
                image_height=gen_params.get("image_height", 1080),
                seed=gen_params.get("seed", 42),
            )

        raise ValueError(f"No construction logic for data_loader '{loader_name}'")

    def random_step(self):
        """Jump to a random sample and run it."""
        if self._state not in (ExperimentState.RUNNING, ExperimentState.FINISHED):
            raise RuntimeError("No experiment running; call start() first")
        assert self._data_loader is not None
        total = len(self._data_loader)
        if total == 0:
            return
        self._current_step = random.randint(0, total - 1)
        self.step()

    def step(self):
        """Advance the running experiment by one step (GUI single-step)."""
        if self._state not in (ExperimentState.RUNNING, ExperimentState.FINISHED):
            raise RuntimeError("No experiment running; call start() first")
        if self._waiting_for_response:
            return

        # Allow stepping again after FINISHED (re-entering from last step)
        if self._state == ExperimentState.FINISHED:
            self._state = ExperimentState.RUNNING

        # Check bounds
        assert self._data_loader is not None
        if self._current_step >= len(self._data_loader):
            log.info("All samples processed")
            self._state = ExperimentState.FINISHED
            mark_experiment_finished()
            return

        if not self._setup_step():
            return

        fut = self._begin_chain()
        fut.add_done_callback(self._on_single_step_done)

    def _setup_step(self) -> bool:
        """Populate context for the current sample (no inference request).

        Returns True if the sample is ready to run, False on a setup failure
        (the step is aborted and overlay cleared).
        """
        # Clear interpreter variables so previous step's state doesn't leak
        hfi.variables.clear()

        # Reset the hybrid agent service so no turn count, tool-call dedup
        # state, or lingering continuation chain leaks into this sample.
        vii.project_manager.hybrid_segment_service.reset_turns()

        # Emit progress
        current, total = self.progress
        vii.app.view_state.overlay_VS.progress_changed.emit(current, total)
        if self._app_view_model is not None:
            self._app_view_model.experiment_progress.emit(current, total)

        assert self._data_loader is not None
        try:
            sample = self._data_loader.get_sample(self._current_step)
        except Exception:
            log.exception("Error getting sample from data loader")
            self._abort_step("Data loader error")
            return False

        # Prepare overlay display (GUI only)
        if self._app_view_model is not None:
            self._display_sample(sample.image, sample.ground_truth_shapes)

        # Extract GT bbox in 0-1000 xyxy format for IoU comparison
        self._current_sample_gt_bbox = None
        for shape in sample.ground_truth_shapes:
            if shape["type"] == "rect":
                x, y, w, h = shape["geometry"]
                self._current_sample_gt_bbox = (x, y, x + w, y + h)
                break  # Use first rect as GT

        # Populate context (system prompts + prompt + image). The inference
        # request is inserted by the hybrid service's run_chain, not here.
        try:
            self._current_sample_image = sample.image
            assert self._config is not None
            submit_experiment_context(
                image=sample.image,
                prompt=sample.prompt,
                config=self._config,
            )
        except Exception:
            log.exception("Error during experiment step setup")
            self._abort_step("Exception during step setup")
            return False

        return True

    def _begin_chain(self) -> asyncio.Future:
        """Start the agent chain for the current sample and track its future."""
        assert self._config is not None
        request_extras: dict = {
            "generation_params": self._model_gen_params(),
            "chat_template_params": self._config.chat_template_params,
        }
        if not self._config.stream:
            request_extras["stream"] = False
        fut = vii.project_manager.hybrid_segment_service.run_chain(
            self._config.focus_mode, request_extras=request_extras
        )
        self._step_future = fut
        self._waiting_for_response = True
        return fut

    def _model_gen_params(self) -> dict:
        """Filter config generation params to model-relevant keys only."""
        assert self._config is not None
        MODEL_GEN_KEYS = {
            "max_new_tokens",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "do_sample",
        }
        return {
            k: v
            for k, v in self._config.generation_params.items()
            if k in MODEL_GEN_KEYS
        }

    def _run_params_snapshot(self) -> dict:
        """Snapshot the model call parameters used for this experiment run.

        Captured at run start and persisted alongside results so the stats
        report can show exactly how the model was invoked.
        """
        assert self._config is not None
        return {
            "model": vii.get_config().selected_model,
            "generation_params": self._model_gen_params(),
            "chat_template_params": dict(self._config.chat_template_params or {}),
            "stream": self._config.stream,
            "focus_mode": self._config.focus_mode,
            "extraction": self._config.extraction,
        }

    def _on_single_step_done(self, fut: asyncio.Future) -> None:
        """Finalize a GUI single-step once its chain future settles."""
        self._waiting_for_response = False
        if fut.cancelled():
            return
        exc = fut.exception()
        if exc is not None:
            log.error("Single step failed: %s", exc)
            vii.app.view_state.overlay_VS.clear()
            return
        self._on_step_settled(self._last_assistant_text())

    def _last_assistant_text(self) -> str:
        """Return the text of the most recent non-empty assistant message."""
        cm = vii.project_manager.context_manager
        for item in cm.items_reversed():
            if isinstance(item, TextMessage) and item.origin == "assistant":
                text = item.text or ""
                if text.strip():
                    return text
        return ""

    def _display_sample(
        self, image: Image.Image, ground_truth_shapes: list[Shape]
    ) -> None:
        """Compute display transform and update overlay via action."""
        # Convert to QImage for overlay
        buf = BytesIO()
        image.save(buf, format="PNG")
        qimg = QImage()
        qimg.loadFromData(buf.getvalue())

        # Compute display transform: original image → overlay widget
        capture = vii.app.view_state.capture_screen_info
        if not capture:
            return
        from assistant.util import get_screen_by_name

        screen = get_screen_by_name(capture.name)
        if not screen:
            return
        geo = screen.geometry()
        screen_w, screen_h = geo.width(), geo.height()

        img_w, img_h = image.size
        disp_scale = min(screen_w / img_w, screen_h / img_h)
        draw_w = int(img_w * disp_scale)
        draw_h = int(img_h * disp_scale)
        offset_x = (screen_w - draw_w) / 2
        offset_y = (screen_h - draw_h) / 2
        dt = DisplayTransform(
            scale=disp_scale,
            offset_x=offset_x,
            offset_y=offset_y,
            image_width=img_w,
            image_height=img_h,
        )

        # Map ground truth shapes: 0-1000 grid (on original image) → widget pixels
        scaled_gt: list[Shape] = []
        for shape in ground_truth_shapes:
            if shape["type"] == "rect":
                x, y, w, h = shape["geometry"]
                orig_x = x / 1000 * img_w
                orig_y = y / 1000 * img_h
                orig_w = w / 1000 * img_w
                orig_h = h / 1000 * img_h
                wx, wy, ww, wh = dt.image_rect_to_widget(orig_x, orig_y, orig_w, orig_h)
                scaled_gt.append({**shape, "geometry": (wx, wy, ww, wh)})
            else:
                scaled_gt.append(shape)

        # Compute resize metadata for the action (needed for shape mapping)
        from assistant.image_ops import resize_to_target
        from assistant.model_configs import get_resolution_for_model

        assert self._config is not None
        res_override = (
            tuple(self._config.resolution) if self._config.resolution else None
        )
        model_key = vii.get_config().selected_model
        target_w, target_h = get_resolution_for_model(model_key, override=res_override)
        _, meta = resize_to_target(image, target_w, target_h)

        update_experiment_overlay(
            sample_image=qimg,
            display_transform=dt,
            gt_shapes=scaled_gt,
            resize_meta=meta,
        )

    def stop(self):
        if self._state == ExperimentState.IDLE:
            return

        self._run_all_active = False
        self._cancelled = True
        self._state = ExperimentState.IDLE
        self._current_step = 0
        self._data_loader = None
        self._waiting_for_response = False
        # Cancel the running procedure task if any
        if self._run_all_task and not self._run_all_task.done():
            self._run_all_task.cancel()
            self._run_all_task = None
        # Halt the active chain and reject its settle future
        vii.project_manager.hybrid_segment_service.cancel_chain("experiment stopped")
        # Restore the service's default turn cap (undo any per-experiment override)
        hss = vii.project_manager.hybrid_segment_service
        hss.MAX_TURNS = type(hss).MAX_TURNS
        self._save_results_json()
        clear_experiment_context()
        vii.app.view_state.overlay_VS.clear()
        log.info("Experiment stopped")

    async def _run_one_step(self) -> None:
        """Run a single experiment step and await the chain settle future."""
        if not self._setup_step():
            raise RuntimeError("Experiment step setup failed")

        fut = self._begin_chain()
        chain_result = await asyncio.wait_for(fut, timeout=INFERENCE_TIMEOUT_S)

        # Chain settled — record results and advance the step pointer.
        self._on_step_settled(self._last_assistant_text(), chain_result)

    def _record_error_step(self, error: str) -> None:
        """Record a failed step and save the sample image (GT only)."""
        hss = vii.project_manager.hybrid_segment_service
        result = {
            "step": self._current_step,
            "response": None,
            "predicted_bbox": None,
            "gt_bbox": (
                list(self._current_sample_gt_bbox)
                if self._current_sample_gt_bbox
                else None
            ),
            "iou": 0.0,
            "correct": False,
            "center_in_gt": False,
            "error": error,
            "turns": hss._turn_count,
            "stop_reason": error,
        }
        self._results.append(result)
        # Save image with GT overlay even for errors
        if self._output_dir and self._current_sample_image:
            save_sample_image(self._output_dir, self._current_sample_image, result)
        # Append to JSONL incrementally
        if self._output_dir:
            append_result_jsonl(self._output_dir, result)

    async def _reconnect(self) -> None:
        """Reconnect the WS sync client without re-inserting system prompts.

        Only tears down and re-creates the WebSocket connection. The server
        sends an empty full_state for the new session, which clears the local
        store. The next step() call will insert fresh context items.
        """
        from sivkit.storage.websockets_client_sync import WebSocketsClientSync

        for attempt in range(1, MAX_RECONNECT_ATTEMPTS + 1):
            log.info(
                f"Reconnect attempt {attempt}/{MAX_RECONNECT_ATTEMPTS} "
                f"in {RECONNECT_DELAY_S}s..."
            )
            await asyncio.sleep(RECONNECT_DELAY_S)

            pm = vii.project_manager
            settings_state = vii.app.view_state.settings_VS

            # Tear down old client
            if pm._sync_client:
                pm._sync_client.stop()
                pm._sync_client = None

            # Create a new WS client on the same store (no system prompts)
            ws_url = vii.inference_client.ws_url
            store = pm.context_manager._store
            pm._sync_client = WebSocketsClientSync(ws_url, store, role="receiver")

            try:
                await pm._sync_client.connect(timeout=5.0)
            except Exception as exc:
                log.error(f"Reconnection attempt {attempt} failed: {exc}")
                pm._sync_client = None
                continue

            # Re-wire disconnect callback
            pm._sync_client.done.add_done_callback(lambda _: pm._on_sync_disconnected())
            settings_state.session_state = "started"
            log.info("Reconnected successfully")
            return

        raise ConnectionError(
            f"Failed to reconnect after {MAX_RECONNECT_ATTEMPTS} attempts"
        )

    def _on_step_settled(
        self, response_text: str, chain_result: "ChainResult | None" = None
    ) -> None:
        """Handle step completion — agent chain has settled (no more tool calls).

        Extraction method is determined by config.extraction:
        - "response": parse JSON bbox from the final response text
        - "tool_call": read target['bbox'] from the HFI interpreter state
        """
        self._waiting_for_response = False

        predicted_bbox = None
        assert self._config is not None
        extraction = self._config.extraction

        if extraction == "tool_call":
            predicted_bbox = self._extract_bbox_from_hfi()
        else:
            # "response" — parse from final assistant text
            predicted_bbox = parse_bbox_from_response(response_text)

        # The model emits coords on the Qwen 0-1000 grid of the padded canvas
        # it actually saw. Map them back into the GT's coordinate space
        # (0-1000 linear on the original image) so IoU, saved images and the
        # overlay preview all agree.
        if predicted_bbox is not None:
            predicted_bbox = self._predicted_bbox_to_gt_space(predicted_bbox)

        # Get ground truth bbox from current sample's shapes
        gt_bbox = self._current_sample_gt_bbox
        iou = 0.0
        correct = False
        center_in_gt = False
        if predicted_bbox and gt_bbox:
            iou = compute_iou(predicted_bbox, gt_bbox)
            correct = iou >= IOU_CORRECT_THRESHOLD
            center_in_gt = center_in_box(predicted_bbox, gt_bbox)

        # Show the prediction on the overlay (GUI only) next to the GT box.
        if self._app_view_model is not None and predicted_bbox:
            self._show_predicted_bbox(predicted_bbox)

        result = {
            "step": self._current_step,
            "response": response_text,
            "predicted_bbox": list(predicted_bbox) if predicted_bbox else None,
            "gt_bbox": list(gt_bbox) if gt_bbox else None,
            "iou": round(iou, 4),
            "correct": correct,
            "center_in_gt": center_in_gt,
            "turns": chain_result.turn_count if chain_result else None,
            "stop_reason": chain_result.stop_reason.value if chain_result else None,
        }
        self._results.append(result)

        # Save sample image with overlaid shapes
        self._save_sample_image(result)

        # Append to JSONL incrementally
        if self._output_dir:
            append_result_jsonl(self._output_dir, result)

        # Advance the step pointer
        self._current_step += 1

        if not self._run_all_active:
            self._state = ExperimentState.FINISHED
            mark_experiment_finished()

    def _show_predicted_bbox(self, bbox: tuple[int, int, int, int]) -> None:
        """Overlay the extracted prediction (0-1000 xyxy grid) on the sample.

        Uses the same display transform the GT box uses, so the prediction
        lines up with the displayed image. No-op if no transform is active.
        """
        overlay_vs = vii.app.view_state.overlay_VS
        dt = overlay_vs.display_transform
        if dt is None or self._current_sample_image is None:
            return
        img_w, img_h = self._current_sample_image.size
        x1, y1, x2, y2 = bbox
        orig_x = x1 / 1000 * img_w
        orig_y = y1 / 1000 * img_h
        orig_w = (x2 - x1) / 1000 * img_w
        orig_h = (y2 - y1) / 1000 * img_h
        wx, wy, ww, wh = dt.image_rect_to_widget(orig_x, orig_y, orig_w, orig_h)
        pred_shape: Shape = {
            "type": "rect",
            "geometry": (wx, wy, ww, wh),
            "color": "#FF0000",
        }
        overlay_vs.shapes = [pred_shape]

    def _extract_bbox_from_hfi(self) -> tuple[int, int, int, int] | None:
        """Extract bbox from the HFI interpreter's variable state.

        After the model generates code like:
            target = {}
            target['bbox'] = [x1, y1, x2, y2]
            target['snippet'] = crop_image([x1, y1, x2, y2])

        We read hfi.variables["target"]["bbox"] to get the predicted bbox.
        """
        target = hfi.variables.get("target")
        if isinstance(target, dict):
            bbox = target.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                try:
                    return tuple(int(x) for x in bbox)  # type: ignore[return-value]
                except (ValueError, TypeError):
                    pass

        # Fallback: check if bbox was assigned directly as a variable
        bbox = hfi.variables.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                return tuple(int(x) for x in bbox)  # type: ignore[return-value]
            except (ValueError, TypeError):
                pass

        return None

    def _predicted_bbox_to_gt_space(
        self, bbox: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        """Map a model-predicted bbox from the Qwen 0-1000 padded-canvas grid
        into the ground-truth space (0-1000 linear on the original image).

        Uses the per-sample resize metadata stored by ``submit_experiment_context``
        to undo the center-padding and aspect-preserving scale the model saw.
        Returns the bbox unchanged if metadata or the sample image is missing.
        """
        from assistant.image_ops import qwen_grid_to_original_image

        meta = vii.app.view_state.overlay_VS.resize_meta
        if meta is None or self._current_sample_image is None:
            return bbox
        img_w, img_h = self._current_sample_image.size
        ox1, oy1, ox2, oy2 = qwen_grid_to_original_image(bbox, meta)
        return (
            int(ox1 / img_w * 1000),
            int(oy1 / img_h * 1000),
            int(ox2 / img_w * 1000),
            int(oy2 / img_h * 1000),
        )

    def _save_sample_image(self, result: dict) -> None:
        """Save the sample image with GT and prediction overlaid."""
        if self._output_dir is None or self._current_sample_image is None:
            return
        save_sample_image(self._output_dir, self._current_sample_image, result)

    def _save_results_json(self) -> None:
        """Save accumulated results to a JSON file."""
        if self._output_dir is None or not self._results:
            return
        config_name = self._config.name if self._config else "unknown"
        save_results_json(self._output_dir, config_name, self._agent, self._results)
