"""Context data structures and helpers for the inference server.

ContextItem is the base class for items in the inference context.
TextItem and ImageItem represent the two content types.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Generator, NamedTuple, Optional

import attrs
from fusion.libs.model import Entity, entity_type, load_from_dict
from fusion.storage.change import Change
from PIL import Image

from assistant.inference.context_store import ContextStore


@entity_type
class ContextItem(Entity):
    """Base class for context items. Not instantiated directly."""

    position: int = 0
    size: int = 0
    origin: str = ""  # "user", "assistant", "system", "tool"
    request: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = attrs.Factory(dict)


@entity_type
class TextItem(ContextItem):
    text: str = ""


@entity_type
class ImageItem(ContextItem):
    image_b64: str = ""
    width: int = 0
    height: int = 0

    def resolve_image(self) -> Image.Image:
        if not self.image_b64:
            raise ValueError("ImageItem has no base64 image content")
        raw = base64.b64decode(self.image_b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")


class MessageBatch(NamedTuple):
    messages: list[dict[str, Any]]
    images: list[Image.Image]


CONTEXT_POSITION_STEP = 100


class ContextManager:
    def __init__(self) -> None:
        self._store = ContextStore(
            types_for_cached_type_filtering=(ImageItem, TextItem)
        )

    @property
    def store(self) -> ContextStore:
        return self._store

    def _get_sorted_items(self) -> list[ContextItem]:
        items: list[ContextItem] = [
            entity
            for entity in self._store.find(type=ContextItem)
            if isinstance(entity, ContextItem)
        ]
        items.sort(key=lambda item: (item.position, item.id))
        return items

    def items_sorted(self) -> Generator[ContextItem]:
        yield from self._get_sorted_items()

    def items_reversed(self) -> Generator[ContextItem]:
        yield from reversed(self._get_sorted_items())

    def items_as_qwen_chat_messages(self) -> MessageBatch:
        messages: list[dict[str, Any]] = []
        images: list[Image.Image] = []
        current: Optional[dict[str, Any]] = None
        for item in self.items_sorted():
            # Skip pending request items (empty placeholders awaiting generation)
            if item.request and not item.request.get("result"):
                continue
            role = "assistant" if item.origin == "assistant" else "user"

            if current is None or current["role"] != role:
                if current is not None:
                    messages.append(current)
                current = {"role": role, "content": []}

            if isinstance(item, TextItem):
                current["content"].append({"type": "text", "text": item.text})
            elif isinstance(item, ImageItem):
                current["content"].append({"type": "image"})
                images.append(item.resolve_image())

        if current is not None:
            messages.append(current)
        return MessageBatch(messages=messages, images=images)

    def next_position(self) -> int:
        items = self._get_sorted_items()
        if not items:
            return 0
        return items[-1].position + CONTEXT_POSITION_STEP

    def last_image_item(self) -> Optional[ImageItem]:
        """Most recent image item (any origin)."""
        for item in self.items_reversed():
            if isinstance(item, ImageItem):
                return item
        return None

    def last_screenshot_item(self) -> Optional[ImageItem]:
        """Most recent image with origin='screenshot'."""
        for item in self.items_reversed():
            if isinstance(item, ImageItem) and item.origin == "screenshot":
                return item
        return None

    # --- CRUD helpers ---
    async def apply_change(self, change: Change) -> Change:
        # Map incoming Change to repo operations
        if change.is_create():
            new = load_from_dict(dict(change.forward_component))
            if not isinstance(new, ContextItem):
                raise TypeError("Expected ContextItem for CREATE")
            repo_change = self._store.insert_one(new)
        elif change.is_delete():
            old = self._store.find_one(id=change.entity_id)
            if not isinstance(old, ContextItem):
                raise TypeError("Expected ContextItem for DELETE")
            repo_change = self._store.remove_one(old)
        else:  # update
            existing = self._store.find_one(id=change.entity_id)
            if existing is None:
                raise TypeError("Expected existing ContextItem for UPDATE")
            from fusion.libs.model import dump_to_dict

            updated_dict = {**dump_to_dict(existing), **change.forward_component}
            new = load_from_dict(updated_dict)
            repo_change = self._store.update_one(new)
        return repo_change

    # Server-side helpers to mutate context state
    def insert(self, item: ContextItem) -> Change:
        return self._store.insert_one(item)

    def update(self, item: ContextItem) -> Change:
        return self._store.update_one(item)

    def remove(self, item: ContextItem) -> Change:
        return self._store.remove_one(item)

    def clear(self) -> list[Change]:
        """Remove all context items and return the list of deletion Change objects."""
        changes: list[Change] = []
        for item in self.items_sorted():
            changes.append(self._store.remove_one(item))
        return changes
