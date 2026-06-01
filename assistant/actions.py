from __future__ import annotations

from pathlib import Path
from subprocess import DEVNULL, Popen

from PySide6.QtCore import QUrl
from PySide6.QtGui import QClipboard, QDesktopServices, QGuiApplication

from assistant.facade import raise_on_session_inactive, vii
from assistant.inference.context import TextItem
from assistant.services.ocr import ocr_sync, start_ocr
from assistant.utils.capture_utils import clipboard_image


def ocr_clipboard() -> None:
    """Perform OCR on current clipboard image and update UI + clipboard.

    Side effects:
    - Sets assistant_working during async worker
    - On completion, copies recognized text to both standard & selection clipboards
    - Writes first line notification via system notification if available
    """
    pixmap = clipboard_image()
    if not pixmap:
        msg = "No image found in clipboard for OCR"
        vii.app.terminal_state.output_text = msg
        # Attempt desktop notification mirroring success path UX
        try:  # pragma: no cover (depends on notify-send availability)
            Popen(
                ["notify-send", "Assistant OCR", msg],
                start_new_session=True,
                stdout=DEVNULL,
                stderr=DEVNULL,
            )
        except Exception:
            pass
        return

    settings = vii.app.view_state.settings_VS
    settings.assistant_working = True

    def _finished(text: str):
        settings.assistant_working = False
        notify_text = None
        try:
            cb = QGuiApplication.clipboard()
            cb.setText(text, QClipboard.Mode.Clipboard)
            cb.setText(text, QClipboard.Mode.Selection)
            notify_text = text
        except Exception as e:  # pragma: no cover
            text_local = f"{text}\n(Clipboard copy failed: {e})"
            vii.app.terminal_state.output_text = text_local
        else:
            vii.app.terminal_state.output_text = text
        if notify_text:
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


def pause_or_stop_session(*, new_state: str) -> None:
    """Pause (stop recording) the active session."""
    vii.project_manager.pause_session(new_state=new_state)


def new_session() -> None:
    """Prepare a brand new session directory without starting recording."""
    vii.project_manager.new_session()


def open_sessions_folder() -> None:
    sessions_dir: Path = vii.project_manager.sessions_root
    sessions_dir.mkdir(parents=True, exist_ok=True)
    url = QUrl.fromLocalFile(str(sessions_dir))
    opened = QDesktopServices.openUrl(url)
    if not opened:
        print(f"Failed to open sessions directory: {sessions_dir}")


def add_user_message(text: str) -> None:
    # print(f"[TRACE] add_user_message called with text={text!r}")
    raise_on_session_inactive()

    ctx = vii.project_manager.context_manager
    cleaned = text.strip()
    if cleaned:
        # print(f"[TRACE] add_user_message: creating text item")
        text_item = TextItem()
        text_item.position = ctx.next_position()
        text_item.text = cleaned
        text_item.origin = "user"
        text_item.metadata = {"focus_mode": "main"}
        ctx.insert(text_item)

    request_item = TextItem()
    request_item.position = ctx.next_position()
    request_item.origin = "assistant"
    cfg = vii.get_config()
    generation_params = {"max_new_tokens": cfg.max_new_tokens}
    request_item.request = {
        "stream": True,
        "focus_mode": "main",
        "generation_params": generation_params,
    }
    request_item.metadata = {"focus_mode": "main"}
    # print(f"[TRACE] add_user_message: creating request item")
    ctx.insert(request_item)
    vii.app.view_state.settings_VS.assistant_working = True
    # print(f"[TRACE] add_user_message: done")


def stop_assistant() -> None:
    """Cancel the active generation and stop the assistant's agent loop."""
    settings = vii.app.view_state.settings_VS
    ctx_mgr = vii.project_manager.context_manager

    # Find the active (uncompleted) request item and mark it cancelled
    for item in ctx_mgr.items_reversed():
        if not isinstance(item, TextItem):
            continue
        if not isinstance(item.request, dict):
            continue
        if item.request.get("completed"):
            break  # Past the active request
        # Found the active request — cancel it
        updated = item.copy()
        req = dict(updated.request)
        req["cancelled_by_user"] = True
        req["completed"] = True
        updated.request = req
        ctx_mgr.update(updated)
        break

    # Stop the hybrid segment service agent loop
    hybrid = vii.project_manager.hybrid_segment_service
    hybrid.action_gate.interrupt()
    hybrid._stopped = True

    settings.assistant_working = False
