from __future__ import annotations

import json
from base64 import b64encode
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from fusion.libs.entity.change import Change
from fusion.platform.qt_widgets.utils import qpixmap_to_pil

from assistant.actions import _ensure_session
from assistant.image_ops import resize_like_preprocessor
from assistant.inference.context import ContextItem
from assistant.util import get_logger
from assistant.utils.capture_utils import take_screenshot
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

        self._facade.inference_updates.subscribe(self._on_inference_update)

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

        # Attach image to context
        pil = qpixmap_to_pil(pixmap)
        processed_img, meta = resize_like_preprocessor(
            pil, self._facade.image_preprocessor.image_processor
        )
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

        # Load config and add prompt
        config = self._load_config()
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
            "stream": True,
            "max_new_tokens": config.get("max_new_tokens", 256),
            "temperature": config.get("temperature", 0.0),
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

    def _on_inference_update(self, evt):
        if not self.running or not self._waiting_for_response:
            return

        if not isinstance(evt, Change):
            return
        item = evt.new_state
        if not isinstance(item, ContextItem):
            return
        if not item.request or item.request.get("result") != "success":
            return

        self._waiting_for_response = False
        response_text = item.content.get("text", "")
        self._results.append(
            {
                "step": self._current_step,
                "response": response_text,
            }
        )
        log.info(f"Step {self._current_step} done: {response_text[:100]}")

        # Shapes displayed on overlay automatically by HybridSegmentService

        self._current_step += 1
