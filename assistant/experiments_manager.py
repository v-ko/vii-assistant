from __future__ import annotations

import logging
from base64 import b64encode
from enum import Enum
from io import BytesIO
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage
from sivkit.storage.change import Change
from sivkit.storage.delta import Delta

from assistant.experiments.data_loaders import (
    DATA_LOADER_REGISTRY,
    DataLoader,
    ScreenGrabDataLoader,
)
from assistant.facade import vii
from assistant.image_ops import resize_to_target
from assistant.inference.context import ImageItem, TextItem
from assistant.inference.context_store import is_custom_op
from assistant.model.experiment_config import ExperimentConfig
from assistant.model_configs import get_resolution_for_model
from assistant.view_states.overlay import OverlayMode

log = logging.getLogger(__name__)

INFERENCE_TIMEOUT_MS = 120_000  # 2 minutes


class ExperimentState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    FINISHED = "finished"


class ExperimentsManager:
    def __init__(self):
        self._state = ExperimentState.IDLE
        self._current_step = 0
        self._waiting_for_response = False
        self._pending_request_id: str | None = None
        self._results: list[dict] = []
        self._config: ExperimentConfig | None = None
        self._data_loader: DataLoader | None = None
        self.config_path: Path | None = None
        self._timeout_timer = QTimer()
        self._timeout_timer.setSingleShot(True)
        self._timeout_timer.timeout.connect(
            lambda: self._abort_step("Inference timed out")
        )

        # Register for context store changes (inference results)
        store = vii.project_manager.context_manager._store
        store.add_on_changes_callback(self._on_context_store_changed)

    def start(self, config: ExperimentConfig) -> None:
        """Start a new experiment run with the given config."""
        if self._state == ExperimentState.RUNNING:
            raise RuntimeError("Experiment already running; stop it first")
        self._config = config
        self._current_step = 0
        self._results = []
        self._data_loader = self._create_data_loader(config)
        self._state = ExperimentState.RUNNING
        vii.app.view_state.overlay_VS.mode = OverlayMode.EXPERIMENT
        log.info(f"Started experiment: {config.name}")

    @property
    def state(self) -> ExperimentState:
        return self._state

    @property
    def running(self) -> bool:
        return self._state in (ExperimentState.RUNNING, ExperimentState.FINISHED)

    def _abort_step(self, reason: str):
        """Reset experiment state after an error so the overlay is restored."""
        log.error(f"Experiment step aborted: {reason}")
        self._timeout_timer.stop()
        self._waiting_for_response = False
        self._pending_request_id = None
        vii.app.view_state.overlay_VS.clear()
        self._state = ExperimentState.IDLE

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

        raise ValueError(f"No construction logic for data_loader '{loader_name}'")

    def step(self):
        """Advance the running experiment by one step."""
        if self._state not in (ExperimentState.RUNNING, ExperimentState.FINISHED):
            raise RuntimeError("No experiment running; call start() first")
        if self._waiting_for_response:
            log.warning("Still waiting for response from previous step")
            return

        # Allow stepping again after FINISHED (re-entering from last step)
        if self._state == ExperimentState.FINISHED:
            self._state = ExperimentState.RUNNING

        overlay_vs = vii.app.view_state.overlay_VS
        log.info(f"Step {self._current_step}")

        # Clear previous shapes
        overlay_vs.shapes = []
        overlay_vs.dimmed = False

        # Check bounds
        assert self._data_loader is not None
        if self._current_step >= len(self._data_loader):
            log.info("All samples processed")
            self._state = ExperimentState.FINISHED
            overlay_vs.dimmed = True
            return

        try:
            sample = self._data_loader.get_sample(self._current_step)
        except Exception:
            log.exception("Error getting sample from data loader")
            self._abort_step("Data loader error")
            return

        # Display sample image on overlay

        buf = BytesIO()
        sample.image.save(buf, format="PNG")
        qimg = QImage()
        qimg.loadFromData(buf.getvalue())
        overlay_vs.sample_image = qimg

        # Display ground truth shapes
        overlay_vs.gt_shapes = sample.ground_truth_shapes

        try:
            self._submit_step(sample.image, sample.prompt)
        except Exception:
            log.exception("Error during experiment step setup")
            self._abort_step("Exception during step setup")
            return

    def _submit_step(self, image: Image.Image, prompt: str):
        self._clear_context()
        ctx = vii.project_manager.context_manager

        assert self._config is not None
        config = self._config

        # Resolve target resolution (model default or experiment override)
        res_override = None
        if config.resolution:
            res_override = tuple(config.resolution)

        model_key = vii.get_config().selected_model
        target_w, target_h = get_resolution_for_model(model_key, override=res_override)
        processed_img, meta = resize_to_target(image, target_w, target_h)
        buf = BytesIO()
        processed_img.save(buf, format="PNG")
        encoded = b64encode(buf.getvalue()).decode("ascii")

        img_item = ImageItem()
        img_item.position = ctx.next_position()
        img_item.image_b64 = encoded
        img_item.width = meta["width"]
        img_item.height = meta["height"]
        img_item.size = meta["width"] * meta["height"]
        img_item.origin = "experiment"
        ctx.insert(img_item)

        # Add prompt
        text_item = TextItem()
        text_item.position = ctx.next_position()
        text_item.text = prompt
        text_item.origin = "user"
        ctx.insert(text_item)

        # Trigger inference
        request_item = TextItem()
        request_item.position = ctx.next_position()
        request_item.origin = "assistant"
        request_item.request = {
            "stream": config.stream,
            "generation_params": config.generation_params,
            "chat_template_params": config.chat_template_params,
        }
        ctx.insert(request_item)

        self._pending_request_id = request_item.id
        self._waiting_for_response = True
        self._timeout_timer.start(INFERENCE_TIMEOUT_MS)

    def stop(self):
        if self._state == ExperimentState.IDLE:
            return

        self._state = ExperimentState.IDLE
        self._current_step = 0
        self._data_loader = None
        self._timeout_timer.stop()
        self._waiting_for_response = False
        self._pending_request_id = None
        self._clear_context()
        vii.app.view_state.overlay_VS.clear()
        log.info("Experiment stopped")

    def _clear_context(self):
        vii.project_manager.context_manager.clear()

    def _on_context_store_changed(
        self, delta: Delta, origin: str | None = None
    ) -> None:
        """Check for completed inference on the pending request."""
        if not self.running or not self._waiting_for_response:
            return

        for key, change_data in delta.asdict().items():
            if is_custom_op(key):
                continue
            change = Change(*change_data)
            if change.entity_id != self._pending_request_id:
                continue
            if not change.forward_component:
                continue
            request = change.forward_component.get("request")
            if not isinstance(request, dict):
                continue
            result = request.get("result")
            if not result:
                continue  # still in progress

            if result != "success":
                self._abort_step(f"Inference failed with result={result}")
                return

            # Fetch full entity to get the complete response text
            assert self._pending_request_id is not None
            entity = vii.project_manager.context_manager.store.item(
                self._pending_request_id
            )
            response_text = (
                entity.text if entity and isinstance(entity, TextItem) else ""
            )
            self._on_inference_complete(response_text)
            return

    def _on_inference_complete(self, response_text: str) -> None:
        """Handle a successfully completed inference response."""
        self._timeout_timer.stop()
        self._waiting_for_response = False
        self._pending_request_id = None

        self._results.append(
            {
                "step": self._current_step,
                "response": response_text,
            }
        )
        log.info(f"Step {self._current_step} done: {response_text[:100]}")

        # Shapes displayed on overlay automatically by HybridSegmentService

        self._current_step += 1
        self._state = ExperimentState.FINISHED
        vii.app.view_state.overlay_VS.dimmed = True
