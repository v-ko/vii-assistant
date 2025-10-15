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
from typing import Any, NamedTuple, Optional, cast

from fusion.libs.entity import Entity, entity_type
from fusion.libs.entity.change import Change
from fusion.storage.in_memory_repository import InMemoryRepository
from PIL import Image


@entity_type
class ContextItem(Entity):
    position: int = 0
    size: int = 0
    content: dict[str, Any] = field(default_factory=dict)
    request: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None

    def content_kind(self) -> "ContentKind":
        keys = {k for k in ("tool_call", "image", "text") if k in self.content}
        if len(keys) != 1:
            raise ValueError(
                "ContextItem.content must contain exactly one of 'text', 'image',"
                f" 'tool_call'; got {sorted(keys)}"
            )
        if "tool_call" in keys:
            return ContentKind.TOOL_CALL
        if "image" in keys:
            return ContentKind.IMAGE
        return ContentKind.TEXT

    def resolve_image(self) -> Image.Image:
        b64 = self.content.get("image")
        if not isinstance(b64, str):
            raise ValueError("ContextItem has no base64 image content")
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")


class MessageBatch(NamedTuple):
    messages: list[dict[str, Any]]
    images: list[Image.Image]


class ContentKind(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    TOOL_CALL = "tool_call"


CONTEXT_POSITION_STEP = 100


class ContextManager:
    def __init__(self) -> None:
        self._repo = InMemoryRepository(types_for_cached_type_filtering=(ContextItem,))

    def _sorted_items(self) -> list[ContextItem]:
        items: list[ContextItem] = []
        for entity in self._repo.find():
            if isinstance(entity, ContextItem):
                items.append(entity)
        items.sort(key=lambda item: (item.position, item.id))
        return items

    def list_items(self) -> list[ContextItem]:
        return [cast(ContextItem, item.copy()) for item in self._sorted_items()]

    def items_as_qwen_chat_messages(self) -> MessageBatch:
        items = self._sorted_items()
        messages: list[dict[str, Any]] = []
        images: list[Image.Image] = []
        current: Optional[dict[str, Any]] = None

        for item in items:
            kind = item.content_kind()
            origin = (item.metadata or {}).get("origin")
            role = (
                "assistant"
                if origin == "assistant" or kind is ContentKind.TOOL_CALL
                else "user"
            )

            if kind is ContentKind.TOOL_CALL:
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

            if kind is ContentKind.TEXT:
                text = item.content.get("text", "")
                current["content"].append({"type": "text", "text": str(text)})
            elif kind is ContentKind.IMAGE:
                current["content"].append({"type": "image"})
                images.append(item.resolve_image())
            else:
                raise RuntimeError(f"Unhandled content kind: {kind}")

        if current is not None:
            messages.append(current)
        return MessageBatch(messages=messages, images=images)

    def next_position(self) -> int:
        items = self._sorted_items()
        if not items:
            return 0
        return items[-1].position + CONTEXT_POSITION_STEP

    # --- CRUD helpers ---
    async def apply_change(self, change: Change) -> Change:
        # Map incoming Change to repo operations
        if change.is_create():
            new = change.new_state
            if not isinstance(new, ContextItem):
                raise TypeError("Expected ContextItem for CREATE")
            repo_change = self._repo.insert_one(new)
        elif change.is_delete():
            old = change.old_state
            if not isinstance(old, ContextItem):
                raise TypeError("Expected ContextItem for DELETE")
            repo_change = self._repo.remove_one(old)
        else:  # update
            new = change.new_state
            if not isinstance(new, ContextItem):
                raise TypeError("Expected ContextItem for UPDATE")
            repo_change = self._repo.update_one(new)
        return repo_change

    # Server-side helpers to mutate context state
    def insert(self, item: ContextItem) -> Change:
        return self._repo.insert_one(item)

    def update(self, item: ContextItem) -> Change:
        return self._repo.update_one(item)

    def remove(self, item: ContextItem) -> Change:
        return self._repo.remove_one(item)

    def clear(self) -> list[Change]:
        """Remove all context items and return the list of deletion Change objects.

        The caller is responsible for propagating these deletions to any view state
        or downstream consumers (e.g., pushing over channels). This keeps the
        manager ignorant of UI concerns.
        """
        changes: list[Change] = []
        # Collect current items (already sorted for deterministic removal order)
        for item in self._sorted_items():
            changes.append(self._repo.remove_one(item))
        return changes


# Note: serialization/deserialization helpers removed; prefer fusion.libs.entity API directly
