"""Actions for the VII assistant terminal window.

All state mutations go through @action-decorated functions.
The QML ViewModel calls these — it never mutates state directly.
"""

from __future__ import annotations

import logging
from base64 import b64encode
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from sivkit.libs.action import action
from sivkit.libs.procedure import procedure

from assistant.actions import add_user_message, ocr_clipboard
from assistant.facade import raise_on_session_inactive, vii
from assistant.image_ops import resize_to_target
from assistant.inference.context import ImageMessage
from assistant.inference.focus_modes import PERCEPTION_MODES
from assistant.model_configs import (
    MODEL_SPECS,
    get_resolution_for_model,
)
from assistant.services.segment_parsing import hfi
from assistant.util import get_screen_by_name
from assistant.utils.capture_utils import clipboard_image, take_screenshot
from assistant.utils.image_utils import qpixmap_to_pil

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


def _ensure_model_loaded() -> None:
    state = vii.app.view_state.inference_status_VS.model_state
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

    item = ImageMessage()
    item.position = ctx.next_position()
    item.image_b64 = encoded
    item.width = meta["width"]
    item.height = meta["height"]
    item.size = meta["width"] * meta["height"]
    item.origin = source
    item.metadata = {"visible_to": list(PERCEPTION_MODES)}
    ctx.insert(item)


# ── Settings actions ─────────────────────────────────────────────


@action("terminal.set_screen")
def set_screen(screen_name: str) -> None:
    cfg = vii.get_config()
    if cfg.capture_screen == screen_name:
        return
    cfg.capture_screen = screen_name
    vii.update_config(cfg)


@action("terminal.persist_selected_model")
def persist_selected_model(model_key: str) -> None:
    """Persist the selected model to config (projector updates the VS)."""
    cfg = vii.get_config()
    cfg.selected_model = model_key
    vii.update_config(cfg)


@procedure
async def set_model(model_key: str) -> None:
    """Persist the selection, then ask the server to (un)load the model.

    The client updates its status attributes, which the projector copies
    onto inference_status_VS — the view updates from there.
    """
    persist_selected_model(model_key)
    await vii.inference_client.load_model(model_key)


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
    agent = vii.active_agent
    if not agent:
        return
    prompt_path = mode_config.system_prompt_path(agent)
    # Create file if it doesn't exist
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    if not prompt_path.exists():
        prompt_path.write_text("", encoding="utf-8")
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(prompt_path)))


# ── Experiment actions ───────────────────────────────────────────


@action("terminal.step_experiment")
def step_experiment() -> None:
    _ensure_model_loaded()
    em = vii.experiments_manager
    ensure_experiment_started()
    em.step()


def ensure_experiment_started() -> None:
    """Auto-start the selected experiment config if not already running.

    Does NOT step/begin a chain — only loads the config and starts the
    experiment. Safe to call before run-all (which drives stepping itself).
    """
    em = vii.experiments_manager
    if em.running:
        return
    config_path = em.config_path
    log.info(f"step_experiment: auto-starting from config_path={config_path}")
    if config_path is None or not config_path.exists():
        raise RuntimeError(f"No experiment config selected (path={config_path})")
    import json

    from assistant.model.experiment_config import ExperimentConfig

    data = json.loads(config_path.read_text())
    config = ExperimentConfig(
        id=ExperimentConfig.id_for_path(config_path.stem),
        name=data.get("name", config_path.stem),
        path=str(config_path),
        data_loader=data.get("data_loader", ""),
        prompt=data.get("prompt", ""),
        dataset_path=data.get("dataset_path", ""),
        prompt_template=data.get("prompt_template", ""),
        generation_params=data.get("generation_params", {}),
        resolution=data.get("resolution"),
        start_index=data.get("start_index"),
        end_index=data.get("end_index"),
        stream=data.get("stream", True),
        chat_template_params=data.get("chat_template_params", {}),
        extraction=data.get("extraction", "response"),
        focus_mode=data.get("focus_mode", "main"),
        max_turns=data.get("max_turns") or {},
    )
    log.info(
        f"step_experiment: starting '{config.name}' data_loader={config.data_loader}"
    )
    em.start(config)


@action("terminal.stop_experiment")
def stop_experiment() -> None:
    vii.experiments_manager.stop()


@action("terminal.cancel_experiment")
def cancel_experiment() -> None:
    vii.experiments_manager.cancel()


@action("terminal.random_experiment_step")
def random_experiment_step() -> None:
    _ensure_model_loaded()
    em = vii.experiments_manager
    if not em.running:
        step_experiment()  # auto-start first
    em.random_step()


@action("terminal.open_experiment_config")
def open_experiment_config() -> None:
    config_path = vii.experiments_manager.config_path
    if config_path and config_path.exists():
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(config_path)))


@action("terminal.generate_experiment_stats")
def generate_experiment_stats() -> None:
    """Generate stats from the current experiment's results and open the report."""
    em = vii.experiments_manager
    if em._output_dir is None or not em._output_dir.exists():
        log.warning("No experiment output directory available")
        return

    from assistant.experiments.stats import (
        compute_stats,
        format_stats,
        load_results,
        load_run_params,
    )

    results = load_results(em._output_dir)
    if not results:
        log.warning("No results found to compute stats from")
        return

    stats = compute_stats(results)
    run_params = load_run_params(em._output_dir)
    report = format_stats(em._output_dir, stats, run_params)

    # Write report to file and open it
    report_path = em._output_dir / "stats.txt"
    report_path.write_text(report)
    log.info(f"Stats report: {report_path}\n{report}")
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(report_path)))


# ── Settings modal visibility ─────────────────────────────────────


@action("settings_modal.set_visible")
def set_settings_modal_visible(visible: bool) -> None:
    vii.app.view_state.settings_modal_VS.visible = visible


# ── Toggle terminal ──────────────────────────────────────────────


@action("set_terminal_visible")
def set_terminal_visible(visible: bool) -> None:
    vii.app.terminal_state.visible = visible


@action("terminal.toggle_terminal")
def toggle_terminal() -> None:
    terminal_state = vii.app.terminal_state
    set_terminal_visible(not terminal_state.visible)
