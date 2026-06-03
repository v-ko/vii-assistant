from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QMessageBox
from sivkit.libs.procedure import procedure
from sivkit.logging import get_logger
from sivkit.storage.delta import Delta

if TYPE_CHECKING:
    from assistant.services.hybrid_segment_service import HybridSegmentService

from assistant.facade import vii
from assistant.inference.context import TextItem
from assistant.terminal_actions import show_context_debug

log = get_logger(__name__)


@procedure
async def handle_hybrid_context_delta(
    hybrid_segment_service: HybridSegmentService,
    delta: Delta,
    origin: str | None = None,
) -> None:
    try:
        changed_items = hybrid_segment_service.changed_context_items(delta)

        for item in changed_items:
            if isinstance(item, TextItem) and item.request is not None:
                hybrid_segment_service.update_overlay_from_text(item.text)

        for item in changed_items:
            if not isinstance(item, TextItem):
                continue
            if not isinstance(item.request, dict):
                continue

            # Client-side tool execution (python, click_at, scroll)
            if item.request.get("execution") == "client":
                log.info(
                    "Delta dispatch: execution=client item=%s focus_mode=%s",
                    item.id,
                    item.request.get("focus_mode"),
                )
                await hybrid_segment_service.process_client_execution_request(item)
                continue

            # Completed assistant messages — parse inline tool calls
            if item.request.get("completed"):
                await hybrid_segment_service.process_completed_assistant_message(item)
    except Exception:
        log.error("Hybrid segment context handling failed", exc_info=True)


@procedure
async def fetch_raw_context_and_present(focus_mode: str) -> None:
    """Fetch formatted context from the inference server and show it."""

    try:
        text = await vii.inference_client.get_context_debug(focus_mode)
    except Exception as exc:
        log.error("Failed to fetch context debug for %r", focus_mode, exc_info=True)
        QMessageBox.warning(None, "Context fetch failed", str(exc))
        return
    show_context_debug(focus_mode, text)
