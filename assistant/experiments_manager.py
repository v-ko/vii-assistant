from __future__ import annotations

import json
from base64 import b64encode
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from fusion.storage.change import Change

from assistant.actions import _ensure_session
from assistant.image_ops import resize_to_target
from assistant.inference.context import ContextItem
from assistant.model_configs import get_resolution_for_model
from assistant.util import get_logger
from assistant.utils.capture_utils import take_screenshot
from assistant.utils.image_utils import qpixmap_to_pil
from assistant.view_states.overlay import OverlayMode

if TYPE_CHECKING:
    from assistant.facade import Facade

log = get_logger(__name__)

EXPERIMENTS_DIR = Path(__file__).parent / "experiments"
DEFAULT_EXPERIMENT_CONFIG = EXPERIMENTS_DIR / "element_localization.json"


class ExperimentState(Enum):
    IDLE = "idle"
    RUNNING = "running"


class ExperimentsManager:
    def __init__(self, facade_ref: Facade):
        self._facade = facade_ref
        self._state = ExperimentState.IDLE
        self._current_step = 0
        self._waiting_for_response = False
        self._results: list[dict] = []
        self._config: dict = {}
        self._config_path: Path = DEFAULT_EXPERIMENT_CONFIG

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
        return self._state == ExperimentState.RUNNING

    def step(self):
        if self._waiting_for_response:
            log.warning("Still waiting for response from previous step")
            return

        overlay_vs = self._facade.app_state.overlay_VS

        if self._state == ExperimentState.IDLE:
            self._state = ExperimentState.RUNNING
            overlay_vs.mode = OverlayMode.EXPERIMENT

        log.info(f"Step {self._current_step}")

        # Clear previous context and shapes
        overlay_vs.shapes = []

        # Take screenshot as the sample
        screen = self._facade.current_watched_screen()
        pixmap = take_screenshot(screen)
        if pixmap is None:
            log.error("Failed to capture screenshot")
            return

        # Set sample image on overlay
        overlay_vs.sample_image = pixmap.toImage()

        _ensure_session()
        self._clear_context()
        controller = self._facade.context_controller

        # Load config first — we need resolution info for preprocessing
        config = self._load_config()

        # Attach image to context
        pil = qpixmap_to_pil(pixmap)

        # Resolve target resolution (model default or experiment override)
        res_override = None
        if config.get("resolution"):
            res_override = tuple(config["resolution"])

        model_key = self._facade.app_state.settings_VS.selected_model
        target_w, target_h = get_resolution_for_model(model_key, override=res_override)
        processed_img, meta = resize_to_target(pil, target_w, target_h)
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
            text=config["prompt"],
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

        self._waiting_for_response = True

        # TODO: display ground truth (no-op for now)

    def stop(self):
        if not self.running:
            return

        self._state = ExperimentState.IDLE
        self._current_step = 0
        self._waiting_for_response = False
        self._clear_context()
        self._facade.app_state.overlay_VS.clear()
        log.info("Experiment stopped")

    def _clear_context(self):
        self._facade.context_controller.clear()

    def _on_inference_update(self, change: Change):
        """Called by facade's on_context_store_changes for each change."""
        if not self.running or not self._waiting_for_response:
            return

        if not change.forward_component:
            return
        # Check if this is a completed inference response
        request = change.forward_component.get("request")
        if not isinstance(request, dict) or request.get("result") != "success":
            return

        self._waiting_for_response = False
        content = change.forward_component.get("content", {})
        response_text = content.get("text", "") if isinstance(content, dict) else ""
        self._results.append(
            {
                "step": self._current_step,
                "response": response_text,
            }
        )
        log.info(f"Step {self._current_step} done: {response_text[:100]}")

        # Shapes displayed on overlay automatically by HybridSegmentService

        self._current_step += 1
