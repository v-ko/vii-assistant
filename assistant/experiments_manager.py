from __future__ import annotations

import json
from base64 import b64encode
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from fusion.storage.change import Change
from PIL import Image
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage

from assistant.actions import _ensure_session
from assistant.experiments.data_loaders import (
    DATA_LOADER_REGISTRY,
    DataLoader,
    ScreenGrabDataLoader,
)
from assistant.image_ops import resize_to_target
from assistant.inference.context import ContextItem
from assistant.model_configs import get_resolution_for_model
from assistant.util import get_logger
from assistant.view_states.overlay import OverlayMode

if TYPE_CHECKING:
    from assistant.facade import Facade

log = get_logger(__name__)

EXPERIMENTS_DIR = Path(__file__).parent / "experiments"
DEFAULT_EXPERIMENT_CONFIG = EXPERIMENTS_DIR / "element_localization.json"
INFERENCE_TIMEOUT_MS = 120_000  # 2 minutes


class ExperimentState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    FINISHED = "finished"


class ExperimentsManager:
    def __init__(self, facade_ref: Facade):
        self._facade = facade_ref
        self._state = ExperimentState.IDLE
        self._current_step = 0
        self._waiting_for_response = False
        self._pending_request_id: str | None = None
        self._results: list[dict] = []
        self._config: dict = {}
        self._data_loader: DataLoader | None = None
        self._config_path: Path = DEFAULT_EXPERIMENT_CONFIG
        self._timeout_timer = QTimer()
        self._timeout_timer.setSingleShot(True)
        self._timeout_timer.timeout.connect(
            lambda: self._abort_step("Inference timed out")
        )

    @property
    def config_path(self) -> Path:
        return self._config_path

    @config_path.setter
    def config_path(self, value: Path) -> None:
        self._config_path = value

    def _load_config(self) -> dict:
        # TODO: load once at experiment start when adding full play mode
        with open(self._config_path) as f:
            self._config = json.load(f)
        log.info(f"Loaded experiment config: {self._config_path.name}")
        return self._config

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
        self._facade.app_state.overlay_VS.clear()
        self._state = ExperimentState.IDLE

    def _create_data_loader(self, config: dict) -> DataLoader:
        """Instantiate the data loader specified in the experiment config."""
        loader_name = config.get("data_loader")
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
                prompt=config["prompt"],
            )

        raise ValueError(f"No construction logic for data_loader '{loader_name}'")

    def step(self):
        if self._waiting_for_response:
            log.warning("Still waiting for response from previous step")
            return

        overlay_vs = self._facade.app_state.overlay_VS

        if self._state in (ExperimentState.IDLE, ExperimentState.FINISHED):
            self._state = ExperimentState.RUNNING
            overlay_vs.mode = OverlayMode.EXPERIMENT
            # Load config and create data loader at experiment start
            self._load_config()
            self._data_loader = self._create_data_loader(self._config)

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
        _ensure_session()
        self._clear_context()
        controller = self._facade.context_controller

        config = self._config

        # Resolve target resolution (model default or experiment override)
        res_override = None
        if config.get("resolution"):
            res_override = tuple(config["resolution"])

        model_key = self._facade.app_state.settings_VS.selected_model
        target_w, target_h = get_resolution_for_model(model_key, override=res_override)
        processed_img, meta = resize_to_target(image, target_w, target_h)
        buf = BytesIO()
        processed_img.save(buf, format="PNG")
        encoded = b64encode(buf.getvalue()).decode("ascii")

        img_item = ContextItem.create_image(
            position=controller.next_position(),
            image_b64=encoded,
            width=meta["width"],
            height=meta["height"],
            origin="experiment",
        )
        controller.create(img_item)

        # Add prompt
        text_item = ContextItem.create_text(
            position=controller.next_position(),
            text=prompt,
            origin="user",
        )
        controller.create(text_item)

        # Trigger inference
        request_item = ContextItem()
        request_item.position = controller.next_position()
        request_item.content = {"text": ""}
        request_item.request = {
            "stream": bool(config.get("stream", True)),
            "generation_params": dict(config.get("generation_params") or {}),
            "chat_template_params": dict(config.get("chat_template_params") or {}),
        }
        request_item.metadata = {"origin": "user"}
        controller.create(request_item)

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
        self._facade.app_state.overlay_VS.clear()
        log.info("Experiment stopped")

    def _clear_context(self):
        self._facade.context_controller.clear()

    def _on_inference_update(self, change: Change):
        """Called by facade's on_context_store_changes for each change."""
        if not self.running or not self._waiting_for_response:
            return

        # Only care about updates to the request entity we're waiting on
        if change.entity_id != self._pending_request_id:
            return

        if not change.forward_component:
            return
        # Check if this is a completed inference response
        request = change.forward_component.get("request")
        if not isinstance(request, dict):
            return

        result = request.get("result")
        if not result:
            return  # still in progress

        if result != "success":
            self._abort_step(f"Inference failed with result={result}")
            return

        self._timeout_timer.stop()
        self._waiting_for_response = False

        # Fetch the full entity from the store (forward_component only has
        # the diff, which may lack content when streaming was used)
        assert self._pending_request_id is not None
        entity = self._facade.context_manager.store.item(self._pending_request_id)
        response_text = entity.content.get("text", "") if entity else ""

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
        self._facade.app_state.overlay_VS.dimmed = True
