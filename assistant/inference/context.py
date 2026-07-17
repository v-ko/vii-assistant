"""Context data structures and helpers for the inference server.

ContextItem is the base class for items in the inference context.
TextItem and ImageItem represent the two content types.
"""

from __future__ import annotations

import base64
import io
from enum import Enum
from typing import Any, Generator, NamedTuple, Optional

import attrs
from PIL import Image
from sivkit.libs.model import Entity, entity_type, load_from_dict
from sivkit.storage.change import Change

from assistant.inference.context_store import ContextStore


@entity_type
class ContextMessage(Entity):
    """Base class for context items. Not instantiated directly."""

    position: int = 0
    size: int = 0
    origin: str = ""  # "user", "assistant", "system", "tool"
    request: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = attrs.Factory(dict)
    # Parent link for cancellation lineage and continuation dedup. Set on every
    # causally-spawned item (assistant turn -> execution request -> tool result
    # -> next generation request). Live-only; retroactive edits may break it.
    previous_message_id: str = ""
    # Stop marker. Client-only authority. The reducer walks previous_message_id;
    # any cancelled ancestor halts the chain.
    cancelled: bool = False
    # Supervision gate enum: "" | "pending" | "pass" | "correct" | "error".
    # Empty in non-supervised modes (never gates). Born "pending" in supervised
    # mode; only the client may move it to a terminal verdict.
    teacher_feedback: str = ""


@entity_type
class TextMessage(ContextMessage):
    text: str = ""


@entity_type
class ImageMessage(ContextMessage):
    image_b64: str = ""
    blob_ref: str = ""
    width: int = 0
    height: int = 0

    def resolve_image(self, blob_store=None) -> Image.Image:
        """Decode the image from blob_ref (preferred) or image_b64 (fallback).

        Args:
            blob_store: Optional BlobStore instance for resolving blob_ref.
        """
        if self.blob_ref and blob_store is not None:
            path = blob_store.get_path(self.blob_ref)
            if path is not None:
                return Image.open(path).convert("RGB")
        if self.image_b64:
            raw = base64.b64decode(self.image_b64)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        raise ValueError("ImageItem has no image content (no blob_ref or image_b64)")


@entity_type
class FeedbackItem(ContextMessage):
    """Supervisor feedback on an agent turn. Invisible to model context."""

    assessment: str = ""  # "correct", "pass", "error"
    correction_text: str = ""
    source_item_id: str = ""  # the agent message being judged
    request_item_id: str = ""  # the retry request triggered by this feedback


class GateDecision(Enum):
    """Supervision gate outcome for a completed turn."""

    CONTINUE = "continue"  # create the continuation
    HOLD = "hold"  # awaiting supervisor verdict — do nothing
    SUPPRESS = "suppress"  # error verdict — client owns recovery, no auto-continue


def gate_state(item: "ContextMessage") -> GateDecision:
    """Pure predicate: should a completed turn's continuation be created?

    Reads only ``teacher_feedback`` (empty in non-supervised modes).
    """
    tf = getattr(item, "teacher_feedback", "") or ""
    if tf == "":
        return GateDecision.CONTINUE  # non-supervised
    if tf == "pending":
        return GateDecision.HOLD
    if tf == "error":
        return GateDecision.SUPPRESS
    # "pass" / "correct" (and any unknown terminal verdict)
    return GateDecision.CONTINUE


class MessageBatch(NamedTuple):
    messages: list[dict[str, Any]]
    images: list[Image.Image]


CONTEXT_POSITION_STEP = 100


class ContextManager:
    def __init__(self) -> None:
        self._store = ContextStore(
            types_for_cached_type_filtering=(ImageMessage, TextMessage)
        )

    @property
    def store(self) -> ContextStore:
        return self._store

    def _get_sorted_items(self) -> list[ContextMessage]:
        items: list[ContextMessage] = [
            entity
            for entity in self._store.find(type=ContextMessage)
            if isinstance(entity, ContextMessage)
        ]
        items.sort(key=lambda item: (item.position, item.id))
        return items

    def items_sorted(self) -> Generator[ContextMessage]:
        yield from self._get_sorted_items()

    def items_reversed(self) -> Generator[ContextMessage]:
        yield from reversed(self._get_sorted_items())

    def items_as_qwen_chat_messages(
        self, focus_mode: str | None = None, blob_store=None
    ) -> MessageBatch:
        messages: list[dict[str, Any]] = []
        images: list[Image.Image] = []
        current: Optional[dict[str, Any]] = None
        for item in self.items_sorted():
            # Skip pending request items (empty placeholders awaiting generation)
            if item.request and not item.request.get("result"):
                continue

            # Focus mode filtering
            if focus_mode is not None:
                item_mode = (item.metadata or {}).get("focus_mode")
                visible_to = (item.metadata or {}).get("visible_to")
                # Items must explicitly declare visibility:
                # - focus_mode matches active mode, OR
                # - visible_to list includes active mode
                # Items with NEITHER focus_mode nor visible_to are invisible.
                if item_mode is not None and item_mode == focus_mode:
                    pass  # included
                elif isinstance(visible_to, list) and focus_mode in visible_to:
                    pass  # included
                else:
                    continue

            if item.origin == "system":
                role = "system"
            elif item.origin == "assistant":
                role = "assistant"
            else:
                role = "user"

            if current is None or current["role"] != role:
                if current is not None:
                    messages.append(current)
                current = {"role": role, "content": []}

            if isinstance(item, TextMessage):
                current["content"].append({"type": "text", "text": item.text})
            elif isinstance(item, ImageMessage):
                current["content"].append({"type": "image"})
                images.append(item.resolve_image(blob_store=blob_store))

        if current is not None:
            messages.append(current)
        return MessageBatch(messages=messages, images=images)

    def next_position(self) -> int:
        items = self._get_sorted_items()
        if not items:
            return 0
        return items[-1].position + CONTEXT_POSITION_STEP

    def last_image_item(self) -> Optional[ImageMessage]:
        """Most recent image item (any origin)."""
        for item in self.items_reversed():
            if isinstance(item, ImageMessage):
                return item
        return None

    def last_screenshot_item(self) -> Optional[ImageMessage]:
        """Most recent image with origin='screenshot'."""
        for item in self.items_reversed():
            if isinstance(item, ImageMessage) and item.origin == "screenshot":
                return item
        return None

    # --- CRUD helpers ---
    async def apply_change(self, change: Change) -> Change:
        # Map incoming Change to repo operations
        if change.is_create():
            new = load_from_dict(dict(change.forward_component))
            if not isinstance(new, ContextMessage):
                raise TypeError("Expected ContextItem for CREATE")
            repo_change = self._store.insert_one(new)
        elif change.is_delete():
            old = self._store.find_one(id=change.entity_id)
            if not isinstance(old, ContextMessage):
                raise TypeError("Expected ContextItem for DELETE")
            repo_change = self._store.remove_one(old)
        else:  # update
            existing = self._store.find_one(id=change.entity_id)
            if existing is None:
                raise TypeError("Expected existing ContextItem for UPDATE")
            from sivkit.libs.model import dump_to_dict

            updated_dict = {**dump_to_dict(existing), **change.forward_component}
            new = load_from_dict(updated_dict)
            repo_change = self._store.update_one(new)
        return repo_change

    # Server-side helpers to mutate context state
    def insert(self, item: ContextMessage) -> Change:
        return self._store.insert_one(item)

    def update(self, item: ContextMessage) -> Change:
        return self._store.update_one(item)

    def remove(self, item: ContextMessage) -> Change:
        return self._store.remove_one(item)

    def clear(self) -> list[Change]:
        """Remove all context items and return the list of deletion Change objects."""
        changes: list[Change] = []
        for item in self.items_sorted():
            changes.append(self._store.remove_one(item))
        return changes
