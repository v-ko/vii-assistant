from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QClipboard, QDesktopServices, QGuiApplication
from PySide6.QtWidgets import QMessageBox

from assistant.facade import facade
from assistant.inference.context import ContextItem
from assistant.services.ocr import ocr_sync, start_ocr
from assistant.utils.capture_utils import clipboard_image


def ocr_clipboard() -> None:
    """Perform OCR on current clipboard image and update UI + clipboard.

    Side effects:
    - Sets request_in_progress during async worker
    - On completion, copies recognized text to both standard & selection clipboards
    - Writes first line notification via system notification if available
    """
    pixmap = clipboard_image()
    if not pixmap:
        msg = "No image found in clipboard for OCR"
        facade.qt_app.terminal_state.output_text = msg
        # Attempt desktop notification mirroring success path UX
        try:  # pragma: no cover (depends on notify-send availability)
            from subprocess import DEVNULL, Popen  # local import

            Popen(
                ["notify-send", "Assistant OCR", msg],
                start_new_session=True,
                stdout=DEVNULL,
                stderr=DEVNULL,
            )
        except Exception:
            pass
        return

    settings = facade.app_state.settings_VS
    settings.request_in_progress = True

    def _finished(text: str):
        settings.request_in_progress = False
        notify_text = None
        try:
            cb = QGuiApplication.clipboard()
            cb.setText(text, QClipboard.Mode.Clipboard)
            cb.setText(text, QClipboard.Mode.Selection)
            notify_text = text
        except Exception as e:  # pragma: no cover
            text_local = f"{text}\n(Clipboard copy failed: {e})"
            facade.qt_app.terminal_state.output_text = text_local
        else:
            facade.qt_app.terminal_state.output_text = text
        if notify_text:
            from subprocess import DEVNULL, Popen  # local import

            snippet = (
                notify_text.strip().splitlines()[0]
                if notify_text.strip()
                else "<No text recognized>"
            )
            if len(snippet) > 120:
                snippet = f"{snippet[:117]}..."
            try:
                Popen(
                    ["notify-send", "Assistant OCR", snippet],
                    start_new_session=True,
                    stdout=DEVNULL,
                    stderr=DEVNULL,
                )
            except Exception:  # pragma: no cover
                pass

    start_ocr(
        pixmap,
        on_finished=_finished,
        on_error=lambda m: _finished(m),
    )


def get_clipboard_ocr_text(lang: str = "eng") -> str:
    pixmap = clipboard_image()
    if not pixmap:
        return "Error: No image in clipboard"
    try:
        return ocr_sync(pixmap, lang=lang)
    except Exception as e:  # pragma: no cover
        return f"Error: OCR failed: {e}"


def start_session(*, screen_name: str | None = None) -> None:
    """Delegate to AutomationService to start a session."""
    facade.project_manager.start_session(screen_name=screen_name)


def pause_or_stop_session(*, new_state: str) -> None:
    """Pause (stop recording) the active session."""
    facade.project_manager.pause_session(new_state=new_state)


def new_session() -> None:
    """Prepare a brand new session directory without starting recording."""
    facade.project_manager.new_session()


def open_sessions_folder() -> None:
    sessions_dir: Path = facade.project_manager.sessions_root
    sessions_dir.mkdir(parents=True, exist_ok=True)
    url = QUrl.fromLocalFile(str(sessions_dir))
    opened = QDesktopServices.openUrl(url)
    if not opened:
        print(f"Failed to open sessions directory: {sessions_dir}")


def _ensure_session() -> None:
    if facade.app_state.settings_VS.session_state != "started":
        start_session(screen_name=facade.app_state.settings_VS.screen)


def add_user_message(text: str) -> None:
    print(f"[TRACE] add_user_message called with text={text!r}")
    _ensure_session()

    # Guard: warn if no model is loaded on the server
    settings_vs = facade.app_state.settings_VS
    model_state = settings_vs.server_model_state
    if model_state != "loaded":
        QMessageBox.warning(
            None,
            "No model loaded",
            "No model is loaded on the inference server.\n"
            "Please select and load a model first.",
        )
        return

    controller = facade.context_controller
    cleaned = text.strip()
    if cleaned:
        print(f"[TRACE] add_user_message: creating text item")
        text_item = ContextItem.create_text(
            position=controller.next_position(),
            text=cleaned,
            origin="user",
        )
        controller.create(text_item)

    request_item = ContextItem()
    request_item.position = controller.next_position()
    request_item.size = 0
    request_item.content = {"text": ""}
    generation_params = facade.config.get("default_generation_params", {})
    request_item.request = {
        "stream": True,
        "generation_params": dict(generation_params or {}),
    }
    request_item.metadata = {"origin": "user"}
    print(f"[TRACE] add_user_message: creating request item")
    controller.create(request_item)
    print(f"[TRACE] add_user_message: done")
