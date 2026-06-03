"""Snippet procedure — handles the result of a screen region capture.

Two modes:
- During transcription: starts inference for image description and appends
  the future to the transcription output buffer (inserted in order).
- No transcription: copies the captured image to the clipboard.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging

from PIL import Image
from PySide6.QtGui import QGuiApplication, QImage
from sivkit.libs.model import dump_to_dict
from sivkit.libs.procedure import procedure

from assistant.facade import vii
from assistant.inference.context import ImageItem, TextItem
from assistant.snippet_actions import show_snippet_overlays
from assistant.utils.image_utils import qimage_to_pil

log = logging.getLogger(__name__)


def start_snippet() -> None:
    """Entry point from the route/hotkey. Shows overlays and waits for selection."""
    show_snippet_overlays(vii.app.view_state)


@procedure
async def handle_snippet_result(image: QImage) -> None:
    """Process the captured snippet image.

    Called by the controller when a region is captured.
    """
    import time

    from assistant.recording_procedures import (
        _active_task,
        _pending_snippets,
        _recording_start_unix,
    )

    transcribing = _active_task is not None and not _active_task.done()
    log.info("handle_snippet_result: transcribing=%s", transcribing)

    if transcribing:
        # Compute recording-relative time for this snippet
        snippet_time = time.time() - _recording_start_unix
        # Create inference task and register with timestamp
        task = asyncio.ensure_future(_get_snippet_description(image))
        _pending_snippets.append((snippet_time, task))
        log.info(
            "Snippet description task appended (t=%.1fs into recording)", snippet_time
        )
    else:
        _copy_to_clipboard(image)


async def _get_snippet_description(image: QImage) -> str:
    """Get description for a snippet image, formatted for insertion into transcription."""
    pil_image = qimage_to_pil(image)
    log.info(
        "_get_snippet_description: calling inference for %dx%d image",
        image.width(),
        image.height(),
    )
    description = await _get_image_description(pil_image)
    if description:
        log.info("_get_snippet_description: got description=%s", description[:80])
        return f"\n\n[screen snippet: {description}]\n\n"
    else:
        log.warning("No description returned for snippet")
        return ""


async def _get_image_description(pil_image: Image.Image) -> str | None:
    """Call the inference endpoint to describe the image."""
    # Encode image to base64
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    img_item = ImageItem()
    img_item.position = 100
    img_item.origin = "user"
    img_item.image_b64 = image_b64
    img_item.width = pil_image.width
    img_item.height = pil_image.height

    text_item = TextItem()
    text_item.position = 150
    text_item.origin = "user"
    text_item.text = "Briefly describe what is shown in this image region. If the image contains only text, output just the text content verbatim."

    req_item = TextItem()
    req_item.position = 200
    req_item.origin = "assistant"
    req_item.request = {"stream": False}

    context_data = [
        dump_to_dict(img_item),
        dump_to_dict(text_item),
        dump_to_dict(req_item),
    ]

    try:
        data = await vii.inference_client.infer(
            context_data,
            {"max_new_tokens": 128, "do_sample": False},
            timeout=30.0,
        )
    except Exception as exc:
        log.error("Snippet /infer call failed: %s", exc)
        return None

    if data.get("status") != "success":
        log.error("Snippet inference error: %s", data.get("error_message"))
        return None

    return data.get("text", "").strip()


def _copy_to_clipboard(image: QImage) -> None:
    """Copy the captured image to the system clipboard."""
    from PySide6.QtGui import QPixmap

    pixmap = QPixmap.fromImage(image)
    clipboard = QGuiApplication.clipboard()
    clipboard.setPixmap(pixmap)
    log.info("Snippet copied to clipboard (%dx%d)", image.width(), image.height())
