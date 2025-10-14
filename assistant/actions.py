from __future__ import annotations

from pathlib import Path
from typing import Optional

from fusion.libs.entity.change import Change
from PySide6.QtCore import QUrl
from PySide6.QtGui import QClipboard, QDesktopServices, QGuiApplication, QScreen

from assistant.inference.context import ContextItem
from assistant.services.ocr import ocr_sync, start_ocr
from assistant.services.session_recorder import (  # noqa: F401 (kept for API compatibility)
    SessionRecorderConfig,
)
from assistant.utils.capture_utils import clipboard_image


def ocr_clipboard(facade) -> None:
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
        return

    settings = facade.app_state.settings
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


def get_clipboard_ocr_text(facade, lang: str = "eng") -> str:
    pixmap = clipboard_image()
    if not pixmap:
        return "Error: No image in clipboard"
    try:
        return ocr_sync(pixmap, lang=lang)
    except Exception as e:  # pragma: no cover
        return f"Error: OCR failed: {e}"


def start_session(facade, *, screen_name: str | None = None) -> None:
    """Delegate to AutomationService to start a session."""
    facade.project_manager.start_session(screen_name=screen_name)


def pause_or_stop_session(facade, *, new_state: str) -> None:
    """Pause (stop recording) the active session."""
    facade.project_manager.pause_session(new_state=new_state)


def new_session(facade) -> None:
    """Prepare a brand new session directory without starting recording."""
    facade.project_manager.new_session()


__all__ = [
    "start_session",
    "pause_or_stop_session",
    "new_session",
    "ocr_clipboard",
    "get_clipboard_ocr_text",
    "handle_message_submitted",
]


def open_sessions_folder(facade) -> None:
    sessions_dir: Path = facade.project_manager.sessions_root
    sessions_dir.mkdir(parents=True, exist_ok=True)
    url = QUrl.fromLocalFile(str(sessions_dir))
    opened = QDesktopServices.openUrl(url)
    if not opened:
        print(f"Failed to open sessions directory: {sessions_dir}")


def handle_message_submitted(facade, text: str) -> None:
    manager = facade.context_manager
    cleaned = text.strip()
    if cleaned:
        text_item = ContextItem()
        text_item.position = manager.next_position()
        text_item.size = 0
        text_item.content = {"text": cleaned}
        text_item.metadata = {"origin": "user"}
        manager.insert(text_item)
        change = Change.CREATE(text_item)
        facade.app_state.context.apply_change(change)
        try:
            facade.project_manager.publish_client_change(change)
        except Exception:
            pass

    request_item = ContextItem()
    request_item.position = manager.next_position()
    request_item.size = 0
    request_item.content = {"text": ""}
    request_item.request = {
        "stream": True,
        "max_new_tokens": 256,
        "temperature": 0.0,
    }
    request_item.metadata = {"origin": "user", "trigger": "manual-send"}
    manager.insert(request_item)
    change = Change.CREATE(request_item)
    facade.app_state.context.apply_change(change)
    try:
        facade.project_manager.publish_client_change(change)
    except Exception:
        pass
