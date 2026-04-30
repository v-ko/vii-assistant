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
from typing import Any, Generator, NamedTuple, Optional, TypedDict

from fusion.libs.model import Entity, entity_type, load_from_dict
from fusion.storage.change import Change
from fusion.storage.in_memory_store import InMemoryStore
from PIL import Image

from assistant.inference.context_store import ContextStore


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
        self._repo = ContextStore(types_for_cached_type_filtering=(ContextItem,))

    @property
    def store(self) -> ContextStore:
        return self._repo

    def _get_sorted_items(self) -> list[ContextItem]:
        items: list[ContextItem] = [
            entity
            for entity in self._repo.find(type=ContextItem)
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
        items = self._get_sorted_items()
        if not items:
            return 0
        return items[-1].position + CONTEXT_POSITION_STEP

    # --- CRUD helpers ---
    async def apply_change(self, change: Change) -> Change:
        # Map incoming Change to repo operations
        if change.is_create():
            new = load_from_dict(dict(change.forward_component))
            if not isinstance(new, ContextItem):
                raise TypeError("Expected ContextItem for CREATE")
            repo_change = self._repo.insert_one(new)
        elif change.is_delete():
            old = self._repo.find_one(id=change.entity_id)
            if not isinstance(old, ContextItem):
                raise TypeError("Expected ContextItem for DELETE")
            repo_change = self._repo.remove_one(old)
        else:  # update
            existing = self._repo.find_one(id=change.entity_id)
            if existing is None:
                raise TypeError("Expected existing ContextItem for UPDATE")
            updated_dict = {**existing.asdict(), **change.forward_component}
            new = ContextItem(
                **{k: v for k, v in updated_dict.items() if k != "type_name"}
            )
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
        """Remove all context items and return the list of deletion Change objects."""
        changes: list[Change] = []
        for item in self.items_sorted():
            changes.append(self._repo.remove_one(item))
        return changes
