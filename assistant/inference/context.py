"""Context data structures and helpers for the inference server.

Overhauled to a single ContextItem that can hold multiple content variants
and an optional request to trigger model inference.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import field
from enum import StrEnum
from typing import Any, Generator, NamedTuple, Optional, TypedDict, cast

from fusion.libs.entity import Entity, entity_type
from fusion.libs.entity.change import Change
from fusion.storage.in_memory_repository import InMemoryRepository
from PIL import Image


class ContextItemMetadata(TypedDict, total=False):
    origin: Optional[str]
    image_size: Optional[dict[str, int]]  # width, height


@entity_type
class ContextItem(Entity):
    position: int = 0
    size: int = 0
    content: dict[str, Any] = field(default_factory=dict)
    request: Optional[dict[str, Any]] = None
    metadata: ContextItemMetadata = field(default_factory=ContextItemMetadata)

    def content_type(self) -> "ContentType":
        if not self.content:
            raise ValueError("ContextItem has empty content")

        if "image" in self.content:
            return ContentType.IMAGE
        if "tool_call" in self.content:
            return ContentType.TOOL_CALL
        if "text" in self.content:
            return ContentType.TEXT
        raise ValueError(
            "ContextItem has unknown content type for content keys:"
            f" {self.content.keys()})"
        )

    def resolve_image(self) -> Image.Image:
        b64 = self.content.get("image")
        if not isinstance(b64, str):
            raise ValueError("ContextItem has no base64 image content")
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    @classmethod
    def create_image(
        cls,
        *,
        position: int,
        image_b64: str,
        width: int,
        height: int,
        origin: str,
    ) -> "ContextItem":
        item = cls()
        item.position = position
        item.size = width * height
        item.content = {"image": image_b64}
        item.metadata = {
            "origin": origin,
            "image_size": {"width": width, "height": height},
        }
        return item

    @classmethod
    def create_text(
        cls,
        *,
        position: int,
        text: str,
        origin: str,
    ) -> "ContextItem":
        item = cls()
        item.position = position
        item.content = {"text": text}
        item.metadata = {"origin": origin}
        return item


class MessageBatch(NamedTuple):
    messages: list[dict[str, Any]]
    images: list[Image.Image]


class ContentType(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    TOOL_CALL = "tool_call"


CONTEXT_POSITION_STEP = 100


class ContextManager:
    def __init__(self) -> None:
        self._repo = InMemoryRepository(types_for_cached_type_filtering=(ContextItem,))
        # Cache of sorted item ids (position, id) for fast listing
        self._sorted_ids: list[str] = []
        # No dirty flag; list rebuilt eagerly on mutations that can affect ordering

    # --- cache internals ---
    def _rebuild_sorted_ids(self) -> None:
        items: list[ContextItem] = []
        for entity in self._repo.find_cached(
            type=ContextItem
        ):  # avoid copy for sort source
            if isinstance(entity, ContextItem):
                items.append(entity)
        items.sort(key=lambda item: (item.position, item.id))
        # Coerce id to str to keep cache homogenous (Entity.id may be tuple for composite keys)
        self._sorted_ids = [str(item.id) for item in items]

    def _sorted_item_ids(self) -> list[str]:
        if not self._sorted_ids:
            self._rebuild_sorted_ids()
        return self._sorted_ids

    def items_sorted(self) -> Generator[ContextItem]:
        for item_id in self._sorted_item_ids():
            for entity in self._repo.find(id=item_id):
                if isinstance(entity, ContextItem):
                    yield cast(ContextItem, entity)
        # return [cast(ContextItem, item.copy()) for item in self._sorted_items()]

    def items_reversed(self) -> Generator[ContextItem]:
        for item_id in reversed(self._sorted_item_ids()):
            for entity in self._repo.find(id=item_id):
                if isinstance(entity, ContextItem):
                    yield cast(ContextItem, entity)

    def items_as_qwen_chat_messages(self) -> MessageBatch:
        messages: list[dict[str, Any]] = []
        images: list[Image.Image] = []
        current: Optional[dict[str, Any]] = None
        for item in self.items_sorted():
            kind = item.content_type()
            origin = (item.metadata or {}).get("origin")
            role = (
                "assistant"
                if origin == "assistant" or kind is ContentType.TOOL_CALL
                else "user"
            )

            if kind is ContentType.TOOL_CALL:
                if current is not None:
                    messages.append(current)
                    current = None
                tool = item.content.get("tool_call") or {}
                messages.append(
                    {
                        "role": role,
                        "content": [
                            {
                                "type": "text",
                                "text": f"<tool_call>{json.dumps(tool)}</tool_call>",
                            }
                        ],
                    }
                )
                continue

            if current is None or current["role"] != role:
                if current is not None:
                    messages.append(current)
                current = {"role": role, "content": []}

            if kind is ContentType.TEXT:
                text = item.content.get("text", "")
                current["content"].append({"type": "text", "text": str(text)})
            elif kind is ContentType.IMAGE:
                current["content"].append({"type": "image"})
                images.append(item.resolve_image())
            else:
                raise RuntimeError(f"Unhandled content kind: {kind}")

        if current is not None:
            messages.append(current)
        return MessageBatch(messages=messages, images=images)

    def next_position(self) -> int:
        # Use cache to find last item's position without copying all
        ids = self._sorted_item_ids()
        if not ids:
            return 0
        # retrieve last entity (already cached) and read position
        last_id = ids[-1]
        for entity in self._repo.find_cached(id=last_id):  # direct cached entity
            if isinstance(entity, ContextItem):
                return entity.position + CONTEXT_POSITION_STEP
        return 0

    # --- CRUD helpers ---
    async def apply_change(self, change: Change) -> Change:
        # Map incoming Change to repo operations
        if change.is_create():
            new = change.new_state
            if not isinstance(new, ContextItem):
                raise TypeError("Expected ContextItem for CREATE")
            repo_change = self._repo.insert_one(new)
            self._rebuild_sorted_ids()
        elif change.is_delete():
            old = change.old_state
            if not isinstance(old, ContextItem):
                raise TypeError("Expected ContextItem for DELETE")
            repo_change = self._repo.remove_one(old)
            # Deletion removes an id; reflect removal without full rebuild if large list
            try:
                self._sorted_ids.remove(str(old.id))
            except ValueError:
                pass
        else:  # update
            new = change.new_state
            if not isinstance(new, ContextItem):
                raise TypeError("Expected ContextItem for UPDATE")
            repo_change = self._repo.update_one(new)
            # Update may change position => rebuild ordering
            self._rebuild_sorted_ids()
        return repo_change

    # Server-side helpers to mutate context state
    def insert(self, item: ContextItem) -> Change:
        change = self._repo.insert_one(item)
        self._rebuild_sorted_ids()
        return change

    def update(self, item: ContextItem) -> Change:
        change = self._repo.update_one(item)
        self._rebuild_sorted_ids()
        return change

    def remove(self, item: ContextItem) -> Change:
        change = self._repo.remove_one(item)
        try:
            self._sorted_ids.remove(str(item.id))
        except ValueError:
            # If id not present (e.g. cache not initialized), fallback to rebuild
            if self._sorted_ids:
                pass
        return change

    def clear(self) -> list[Change]:
        """Remove all context items and return the list of deletion Change objects.

        The caller is responsible for propagating these deletions to any view state
        or downstream consumers (e.g., pushing over channels). This keeps the
        manager ignorant of UI concerns.
        """
        changes: list[Change] = []
        # Collect current items (already sorted for deterministic removal order)
        for item in self.items_sorted():
            changes.append(self._repo.remove_one(item))
        self._sorted_ids = []  # emptied
        return changes
