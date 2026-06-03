"""Actions for the VII assistant terminal window.

All state mutations go through @action-decorated functions.
The QML ViewModel calls these — it never mutates state directly.
"""

from __future__ import annotations

from base64 import b64encode
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from sivkit import get_logger
from sivkit.libs.action import action

from assistant.actions import add_user_message, ocr_clipboard
from assistant.facade import raise_on_session_inactive, vii
from assistant.image_ops import resize_to_target
from assistant.inference.context import ImageItem
from assistant.model.experiment_config import ExperimentConfig
from assistant.model_configs import (
    MODEL_SPECS,
    get_resolution_for_model,
)
from assistant.services.segment_parsing import hfi
from assistant.util import get_screen_by_name
from assistant.utils.capture_utils import clipboard_image, take_screenshot
from assistant.utils.image_utils import qpixmap_to_pil

log = get_logger(__name__)

if TYPE_CHECKING:
    from assistant.view_states.settings import AssistantSettingsViewState


def _ensure_model_loaded() -> None:
    state = vii.app.view_state.settings_VS.server_model_state
    if state != "loaded":
        raise RuntimeError(
            f"Cannot proceed: model is not loaded (state={state!r}). "
            "Load a model via 'Set model' first."
        )


# ── Session / message actions ────────────────────────────────────


@action("terminal.ensure_session")
def ensure_session() -> None:
    """Fire-and-forget session start (returns Task internally)."""
    if vii.app.view_state.settings_VS.session_state != "started":
        capture = vii.app.view_state.capture_screen_info
        screen_name = capture.name if capture else vii.get_config().capture_screen
        vii.project_manager.start_session(screen_name=screen_name)


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
    # Reset stopped state so the assistant can work again
    vii.project_manager.hybrid_segment_service.reset_turns()
    add_user_message(text)


# ── Capture actions ──────────────────────────────────────────────


@action("terminal.attach_screen")
def attach_screen() -> None:
    capture = vii.app.view_state.capture_screen_info
    if capture is None:
        log.warning("Cannot attach screen: no capture screen configured")
        return
    screen = get_screen_by_name(capture.name)
    if screen is None:
        log.warning("Cannot attach screen: screen '%s' not found", capture.name)
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
    raise_on_session_inactive()
    ctx = vii.project_manager.context_manager
    pil = qpixmap_to_pil(pixmap)
    model_key = vii.get_config().selected_model
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

    item = ImageItem()
    item.position = ctx.next_position()
    item.image_b64 = encoded
    item.width = meta["width"]
    item.height = meta["height"]
    item.size = meta["width"] * meta["height"]
    item.origin = source
    ctx.insert(item)


# ── Settings actions ─────────────────────────────────────────────


@action("terminal.set_screen")
def set_screen(screen_name: str) -> None:
    cfg = vii.get_config()
    if cfg.capture_screen == screen_name:
        return
    cfg.capture_screen = screen_name
    vii.update_config(cfg)


@action("terminal.set_model")
def set_model(model_key: str) -> None:
    # Persist to config store (projector updates VS)
    cfg = vii.get_config()
    cfg.selected_model = model_key
    vii.update_config(cfg)

    # Fire the HTTP request in a background thread
    def _on_model_result(connected, model_state, mk):
        vii.app.app_view_model.health_check_done.emit(connected, model_state, mk)

    vii.inference_client.load_model_bg(model_key, _on_model_result)


@action("terminal.set_max_new_tokens")
def set_max_new_tokens(value: int) -> None:
    value = max(1, value)
    cfg = vii.get_config()
    cfg.max_new_tokens = value
    vii.update_config(cfg)


@action("terminal.set_input_device")
def set_input_device(device_id: str) -> None:
    cfg = vii.get_config()
    cfg.input_device = device_id
    vii.update_config(cfg)


@action("terminal.show_context_debug")
def show_context_debug(focus_mode: str, text: str) -> None:
    vii.app.app_view_model.context_debug_ready.emit(focus_mode, text)


@action("terminal.add_tool_prompt")
def add_tool_prompt() -> None:
    tool_prompt = hfi.generate_tool_prompt()
    if not tool_prompt:
        return
    settings = vii.app.view_state.settings_VS
    current = settings.system_prompt_markdown
    separator = "\n\n" if current.strip() else ""
    settings.system_prompt_markdown = current + separator + tool_prompt


@action("terminal.open_app_config")
def open_app_config() -> None:
    from assistant.services.config_file_adapter import CONFIG_FILE

    config_path = CONFIG_FILE
    if config_path.exists():
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(config_path)))


@action("terminal.open_focus_mode_prompt")
def open_focus_mode_prompt(mode_name: str) -> None:
    from assistant.inference.focus_modes import FOCUS_MODES

    mode_config = FOCUS_MODES.get(mode_name)
    if mode_config is None:
        return
    prompt_path = mode_config.system_prompt_path()
    # Create file if it doesn't exist
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    if not prompt_path.exists():
        prompt_path.write_text("", encoding="utf-8")
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(prompt_path)))


# ── Experiment actions ───────────────────────────────────────────


@action("terminal.step_experiment")
def step_experiment() -> None:
    _ensure_model_loaded()
    mgr = vii.experiments_manager
    if not mgr.running:
        selected = vii.app.view_state.settings_VS.selected_experiment_config
        if not selected:
            raise RuntimeError("No experiment config selected in settings")
        config_id = ExperimentConfig.id_for_path(Path(selected).stem)
        config = vii.get_experiment_config(config_id)
        if config is None:
            raise RuntimeError(f"Experiment config not found in store: {config_id}")
        mgr.start(config)
    mgr.step()


@action("terminal.stop_experiment")
def stop_experiment() -> None:
    vii.experiments_manager.stop()


@action("terminal.open_experiment_config")
def open_experiment_config() -> None:
    selected = vii.app.view_state.settings_VS.selected_experiment_config
    if not selected:
        return
    path = Path(selected)
    if path.exists():
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
    else:
        log.error("Experiment config file not found: %s", path)


# ── Execution mode ───────────────────────────────────────────────


@action("settings.set_execution_mode")
def set_execution_mode(mode: str) -> None:
    vii.app.view_state.settings_VS.execution_mode = mode


@action("settings.set_experiment_config")
def set_experiment_config(path: str) -> None:
    vii.app.view_state.settings_VS.selected_experiment_config = path


# ── Settings modal visibility ─────────────────────────────────────


@action("settings_modal.set_visible")
def set_settings_modal_visible(visible: bool) -> None:
    vii.app.view_state.settings_modal_VS.visible = visible


# ── Terminal visibility ───────────────────────────────────────────


@action("set_terminal_visible")
def set_terminal_visible(visible: bool) -> None:
    vii.app.terminal_state.visible = visible


@action("terminal.toggle_terminal")
def toggle_terminal() -> None:
    terminal_state = vii.app.terminal_state
    set_terminal_visible(not terminal_state.visible)


# ── Health check ──


@action("health.apply_result")
def apply_health_result(connected: bool, model_state: str, model_key: str) -> None:
    """Apply health check result to settings state and auto-start session."""
    settings = vii.app.view_state.settings_VS
    prev_state = settings.server_model_state
    settings.server_model_state = model_state
    settings.server_model_key = model_key

    # Auto-start session when model becomes available
    if model_state == "loaded" and prev_state != "loaded":
        if settings.session_state != "started":
            capture = vii.app.view_state.capture_screen_info
            screen_name = capture.name if capture else ""
            vii.project_manager.start_session(screen_name=screen_name)


def schedule_health_check(callback) -> None:
    """Run health check in background thread, call callback with results."""
    vii.inference_client.check_status_bg(callback)
