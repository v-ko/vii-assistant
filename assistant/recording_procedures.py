"""Recording and transcription procedures.

The @procedure functions are the consumer/policy layer.
They consume the async generator from the orchestrator and apply output.
"""

from __future__ import annotations

import asyncio
import time
import wave
from datetime import datetime
from subprocess import DEVNULL, Popen

import numpy as np
from fusion import get_logger
from fusion.libs.procedure import procedure
from PySide6.QtCore import QTimer
from PySide6.QtGui import QGuiApplication

from assistant.constants import MAX_SAVED_RECORDINGS, RECORDINGS_DIR
from assistant.facade import vii

log = get_logger(__name__)

# Module-level state for the running transcription task
_active_task: asyncio.Task | None = None
_chunk_queue: asyncio.Queue | None = None
_recording_start_unix: float = 0.0  # Unix time when recording started
# Pending snippet descriptions: (recording_relative_time, Task[str])
_pending_snippets: list[tuple[float, asyncio.Task]] = []


def toggle_recording() -> None:
    """Toggle recording on/off. Entry point from route/hotkey."""
    global _active_task, _chunk_queue, _recording_start_unix, _pending_snippets

    rec = vii.audio_recording_service

    if rec.is_recording:
        # Stop recording — signal end to orchestrator
        rec.stop()
        # Note: rec.stop() emits the final chunk via chunk_ready before returning,
        # so it's already in the queue. Now send the sentinel.
        if _chunk_queue is not None:
            _chunk_queue.put_nowait(None)  # signal end
        return

    # Disallow starting if a transcription is still running
    if _active_task is not None and not _active_task.done():
        log.warning("Transcription still in progress, cannot start new recording")
        return

    # Start recording and kick off the transcription procedure
    _chunk_queue = asyncio.Queue()
    _pending_snippets = []

    # Connect signals BEFORE start() so we don't miss early data
    rec.chunk_ready.connect(_on_chunk_ready)
    rec.recording_stopped.connect(_on_recording_stopped)
    rec.start()

    if not rec.is_recording:
        log.error("Recording failed to start")
        _on_recording_stopped()
        _chunk_queue = None
        return

    _recording_start_unix = time.time()
    _active_task = asyncio.ensure_future(_transcribe_to_ydotool())
    _active_task.add_done_callback(_log_task_result)
    log.info("Transcription task started")


def _log_task_result(task: asyncio.Task) -> None:
    """Log unhandled exceptions from the transcription task."""
    if task.cancelled():
        log.info("Transcription task was cancelled")
    elif exc := task.exception():
        log.error("Transcription task failed: %s", exc, exc_info=exc)
    else:
        log.info("Transcription task completed normally")


def _on_chunk_ready(audio_int16: np.ndarray, chunk_start_time: float) -> None:
    """Bridge Qt signal to asyncio queue."""
    global _chunk_queue
    if _chunk_queue is not None:
        log.info(
            "Chunk received: start=%.1fs, samples=%d, queuing",
            chunk_start_time,
            len(audio_int16) if audio_int16 is not None else 0,
        )
        _chunk_queue.put_nowait((audio_int16, chunk_start_time))
    else:
        log.warning("Chunk received but no queue available")


def _on_recording_stopped() -> None:
    """Disconnect signals after recording stops."""
    rec = vii.audio_recording_service

    def _disconnect():
        try:
            rec.chunk_ready.disconnect(_on_chunk_ready)
        except (RuntimeError, TypeError):
            pass
        try:
            rec.recording_stopped.disconnect(_on_recording_stopped)
        except (RuntimeError, TypeError):
            pass

    QTimer.singleShot(0, _disconnect)


# When True, transcription output is pasted chunk-by-chunk as it arrives.
# When False, all output is buffered and pasted once when recording stops.
PASTE_INCREMENTALLY = False


async def _transcribe_to_ydotool() -> None:
    """Consume transcription generator and type output via ydotool."""
    global _chunk_queue, _pending_snippets

    orch = vii.transcription_orchestrator
    rec = vii.audio_recording_service

    try:
        async for text_segment in orch.run(_chunk_queue):
            if text_segment:
                if PASTE_INCREMENTALLY:
                    log.info("Got text segment (%d chars), pasting", len(text_segment))
                    _paste_text(text_segment)
                else:
                    log.info(
                        "Got text segment (%d chars), buffering", len(text_segment)
                    )

        # Build final output using word timestamps for snippet insertion
        if not PASTE_INCREMENTALLY:
            full_text = await _build_final_output(orch)
            if full_text:
                log.info("Pasting final output (%d chars)", len(full_text))
                _paste_text(full_text)

        log.info("Transcription loop finished normally")
    except Exception as exc:
        log.error("Transcription failed: %s", exc, exc_info=True)
        _notify_error(str(exc))
    finally:
        _save_recording(rec)
        _chunk_queue = None
        _pending_snippets = []


async def _build_final_output(orch) -> str:
    """Build the final text output, interleaving transcription with snippet descriptions.

    Uses word-level timestamps from the stitched transcription result to insert
    snippet descriptions at the correct positions.
    """
    result = orch.last_result
    if result is None:
        return ""

    # Await all pending snippet descriptions
    resolved_snippets: list[tuple[float, str]] = []
    for rel_time, task in _pending_snippets:
        try:
            description = await task
            if description:
                resolved_snippets.append((rel_time, description))
        except Exception as exc:
            log.error("Snippet description task failed: %s", exc)

    if not resolved_snippets:
        return result.text

    resolved_snippets.sort(key=lambda x: x[0])

    # Build output by interleaving words and snippet descriptions
    words = result.words
    if not words:
        # No word timestamps — just prepend/append descriptions
        parts = [desc for _, desc in resolved_snippets]
        parts.append(result.text)
        return "".join(parts)

    output_parts: list[str] = []
    current_words: list[str] = []
    snippet_idx = 0

    for word in words:
        # Insert any snippets that belong before this word
        while (
            snippet_idx < len(resolved_snippets)
            and resolved_snippets[snippet_idx][0] <= word.start
        ):
            # Flush accumulated words
            if current_words:
                output_parts.append(" ".join(current_words))
                current_words = []
            output_parts.append(resolved_snippets[snippet_idx][1])
            snippet_idx += 1
        current_words.append(word.word)

    # Flush remaining words
    if current_words:
        output_parts.append(" ".join(current_words))

    # Append any remaining snippets after all words
    while snippet_idx < len(resolved_snippets):
        output_parts.append(resolved_snippets[snippet_idx][1])
        snippet_idx += 1

    return "".join(output_parts)


def _paste_text(text: str) -> None:
    """Insert text at cursor via clipboard + ydotool Ctrl+V.

    Saves/restores previous clipboard to avoid polluting clipboard managers.
    """
    clipboard = QGuiApplication.clipboard()
    prev = clipboard.text()
    clipboard.setText(text)

    try:
        # Ctrl+V via ydotool key codes: 29=LCtrl, 47=V
        Popen(
            ["ydotool", "key", "29:1", "47:1", "47:0", "29:0"],
            start_new_session=True,
            stdout=DEVNULL,
            stderr=DEVNULL,
        )
    except FileNotFoundError:
        log.error("ydotool not found — cannot paste transcription output")
    except Exception as exc:
        log.error("ydotool paste failed: %s", exc)

    # Restore previous clipboard after a short delay
    QTimer.singleShot(500, lambda: clipboard.setText(prev))


def _notify_error(message: str) -> None:
    """Send desktop notification about transcription failure."""
    try:
        Popen(
            ["notify-send", "VII Transcription Error", message[:200]],
            start_new_session=True,
            stdout=DEVNULL,
            stderr=DEVNULL,
        )
    except Exception:
        pass


def _save_recording(rec) -> None:
    """Save full recording buffer as WAV, keeping only the last N recordings."""
    audio = rec.full_buffer_int16
    if len(audio) == 0:
        log.warning("No audio to save")
        return

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = RECORDINGS_DIR / f"recording_{timestamp}.wav"

    try:
        with wave.open(str(filepath), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(16000)
            wf.writeframes(audio.tobytes())
        log.info("Saved recording to %s", filepath)
    except Exception as exc:
        log.error("Failed to save recording: %s", exc)
        return

    # Enforce cap: delete oldest files beyond MAX_SAVED_RECORDINGS
    existing = sorted(RECORDINGS_DIR.glob("recording_*.wav"))
    while len(existing) > MAX_SAVED_RECORDINGS:
        oldest = existing.pop(0)
        try:
            oldest.unlink()
            log.debug("Removed old recording: %s", oldest.name)
        except OSError as exc:
            log.warning("Failed to remove old recording %s: %s", oldest.name, exc)
