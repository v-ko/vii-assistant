"""Actions for the VII assistant terminal window.

All state mutations go through @action-decorated functions.
The QML backend calls these — it never mutates state directly.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from base64 import b64encode
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from fusion import get_logger
from fusion.libs.action import action
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices

from assistant.actions import add_user_message, ocr_clipboard, start_session
from assistant.facade import vii
from assistant.image_ops import resize_to_target
from assistant.inference.context import ContextItem
from assistant.model_configs import (
    AVAILABLE_MODELS,
    MODEL_SPECS,
    get_resolution_for_model,
)
from assistant.services.hybrid_segment_service import hfi
from assistant.services.project_manager import INFERENCE_HTTP_BASE
from assistant.utils.capture_utils import clipboard_image, take_screenshot
from assistant.utils.image_utils import qpixmap_to_pil

log = get_logger(__name__)

if TYPE_CHECKING:
    from assistant.view_states.settings import AssistantSettingsViewState


def _ensure_model_loaded() -> None:
    state = vii.app_state.settings_VS.server_model_state
    if state != "loaded":
        raise RuntimeError(
            f"Cannot proceed: model is not loaded (state={state!r}). "
            "Load a model via 'Set model' first."
        )


# ── Session / message actions ────────────────────────────────────


@action("terminal.ensure_session")
def ensure_session() -> None:
    if vii.app_state.settings_VS.session_state != "started":
        start_session(screen_name=vii.app_state.settings_VS.screen)


@action("terminal.new_session")
def new_session() -> None:
    vii.project_manager.new_session()


@action("terminal.open_sessions_folder")
def open_sessions_folder() -> None:
    sessions_dir: Path = vii.project_manager.sessions_root
    sessions_dir.mkdir(parents=True, exist_ok=True)
    url = QUrl.fromLocalFile(str(sessions_dir))
    QDesktopServices.openUrl(url)


@action("terminal.submit_message")
def submit_message(text: str) -> None:
    _ensure_model_loaded()
    add_user_message(text)


# ── Capture actions ──────────────────────────────────────────────


@action("terminal.attach_screen")
def attach_screen() -> None:
    try:
        screen = vii.current_watched_screen()
    except Exception as e:
        log.warning("Cannot attach screen: %s", e)
        return
    pixmap = take_screenshot(screen)
    if pixmap is None:
        log.warning("Failed to capture screenshot")
        return
    _add_image_item(pixmap, "screenshot")


@action("terminal.attach_clipboard")
def attach_clipboard() -> None:
    pixmap = clipboard_image()
    if pixmap is None:
        log.info("No image in clipboard")
        return
    _add_image_item(pixmap, "clipboard")


@action("terminal.ocr_clipboard")
def ocr_clipboard_action() -> None:
    ocr_clipboard()


def _add_image_item(pixmap, source: str) -> None:
    ensure_session()
    controller = vii.context_controller
    pil = qpixmap_to_pil(pixmap)
    model_key = vii.app_state.settings_VS.selected_model
    spec = MODEL_SPECS.get(model_key, {})
    if not spec.get("vision"):
        log.warning(
            "Selected model %r is not a vision model — skipping image",
            model_key,
        )
        return
    target_w, target_h = get_resolution_for_model(model_key)
    processed_img, meta = resize_to_target(pil, target_w, target_h)

    buf = BytesIO()
    processed_img.save(buf, format="PNG")
    encoded = b64encode(buf.getvalue()).decode("ascii")
    if not encoded:
        raise RuntimeError("Image encoding failed")

    item = ContextItem.create_image(
        position=controller.next_position(),
        image_b64=encoded,
        width=meta["width"],
        height=meta["height"],
        origin=source,
    )
    controller.create(item)


# ── Settings actions ─────────────────────────────────────────────


@action("terminal.set_screen")
def set_screen(screen_name: str) -> None:
    settings = vii.app_state.settings_VS
    if settings.screen != screen_name:
        settings.screen = screen_name


@action("terminal.set_model")
def set_model(model_key: str) -> None:
    settings = vii.app_state.settings_VS
    settings.selected_model = model_key
    # Fire the HTTP request in a background thread
    threading.Thread(target=_do_model_load, args=(model_key,), daemon=True).start()


@action("terminal.add_tool_prompt")
def add_tool_prompt() -> None:
    tool_prompt = hfi.generate_tool_prompt()
    if not tool_prompt:
        return
    settings = vii.app_state.settings_VS
    current = settings.system_prompt_markdown
    separator = "\n\n" if current.strip() else ""
    settings.system_prompt_markdown = current + separator + tool_prompt


@action("terminal.open_app_config")
def open_app_config() -> None:
    config_path = vii.config.config_file
    if config_path.exists():
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(config_path)))


# ── Experiment actions ───────────────────────────────────────────


@action("terminal.step_experiment")
def step_experiment() -> None:
    _ensure_model_loaded()
    vii.experiments_manager.step()


@action("terminal.stop_experiment")
def stop_experiment() -> None:
    vii.experiments_manager.stop()


@action("terminal.open_experiment_config")
def open_experiment_config() -> None:
    config_path = vii.experiments_manager.config_path
    if config_path.exists():
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(config_path)))


# ── Toggle terminal ──────────────────────────────────────────────


@action("terminal.toggle")
def toggle_terminal() -> None:
    qt_app = vii.qt_app
    root = qt_app.terminal_window
    if root and root.property("terminalVisible"):
        qt_app.hide_terminal()
        if qt_app.overlay:
            qt_app.overlay.hide()
    else:
        qt_app.show_terminal()
        if qt_app.overlay:
            qt_app.overlay.show()


# ── Health check (background, not an action since no state mutation) ──


def schedule_health_check(callback) -> None:
    """Run health check in background thread, call callback with results."""
    threading.Thread(target=_do_health_check, args=(callback,), daemon=True).start()


def _do_health_check(callback) -> None:
    url = f"{INFERENCE_HTTP_BASE}/status"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
        model_info = data.get("model", {})
        model_state = model_info.get("state", "unknown")
        model_key = model_info.get("model_key") or ""
        callback(True, model_state, model_key)
    except Exception:
        callback(False, "unknown", "")


def _do_model_load(model_key: str) -> None:
    url = f"{INFERENCE_HTTP_BASE}/model"
    if model_key == "none":
        req = urllib.request.Request(url, method="DELETE")
    else:
        payload = json.dumps({"model_key": model_key}).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        model_info = data.get("model", {})
        model_state = model_info.get("state", "unknown")
        mk = model_info.get("model_key") or ""
        # Marshal result back to main thread via the backend signal
        vii.qt_app.qml_backend.health_check_done.emit(True, model_state, mk)
    except Exception:
        _do_health_check(
            lambda connected, ms, mk: vii.qt_app.qml_backend.health_check_done.emit(
                connected, ms, mk
            )
        )
