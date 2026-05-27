from __future__ import annotations

from typing import TYPE_CHECKING

from fusion.libs.procedure import procedure
from fusion.logging import get_logger
from fusion.storage.delta import Delta

if TYPE_CHECKING:
    from assistant.services.hybrid_segment_service import HybridSegmentService

from assistant.inference.context import TextItem

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
            if (
                isinstance(item, TextItem)
                and isinstance(item.request, dict)
                and item.request.get("completed")
            ):
                await hybrid_segment_service.process_completed_assistant_message(item)
    except Exception:
        log.error("Hybrid segment context handling failed", exc_info=True)
        raise
