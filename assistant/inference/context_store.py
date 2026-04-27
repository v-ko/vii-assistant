"""InMemoryStore subclass with custom delta operations for vii-assistant.

Supports the ``text_append`` custom op: streaming text chunks are
represented as delta entries keyed ``<entity_id>@text_append`` so they
travel through the standard WebSocketSyncService wire protocol without
any transport-level changes.

On the receiving end, ``apply_delta`` detects the ``@`` separator,
extracts the op name, and dispatches to the matching patch method
which mutates the cached entity in-place (no pop→copy→reinsert cycle).
"""

from __future__ import annotations

from typing import Any

from fusion.logging import get_logger
from fusion.storage.change import Change
from fusion.storage.delta import Delta, DeltaData
from fusion.storage.in_memory_store import InMemoryStore

log = get_logger(__name__)

# Separator between entity_id and op name in delta keys.
OP_SEP = "@"


class ContextStore(InMemoryStore):
    """InMemoryStore extended with custom delta operations."""

    # ------------------------------------------------------------------
    # text_append — streaming token patch
    # ------------------------------------------------------------------

    def text_append(self, entity_id: str, chunk: str) -> Delta:
        """Append *chunk* to the entity's ``content["text"]`` in-place.

        Returns a Delta with a custom-op key so it can be forwarded via
        the standard sync protocol.
        """
        entity = self._entity_cache.get(entity_id)
        if entity is None:
            raise KeyError(f"text_append: entity {entity_id!r} not in store")

        # Mutate in-place — no pop/reinsert.
        content: dict[str, Any] = getattr(entity, "content", None) or {}
        old_text = content.get("text", "")
        content["text"] = old_text + chunk
        # attrs frozen id prevents setattr, but content is a mutable dict
        # so the in-place mutation is fine; just ensure the reference is set.
        if getattr(entity, "content", None) is not content:
            entity.content = content  # type: ignore[attr-defined]

        # Build the custom-op delta entry.
        op_key = f"{entity_id}{OP_SEP}text_append"
        forward = {"chunk": chunk}
        reverse = {"chunk": chunk}  # reverse is same chunk (for undo: remove it)
        change = Change(entity_id, reverse, forward)
        delta_data: DeltaData = {op_key: list(change.asdict())}
        delta = Delta.from_data(delta_data)

        if self.on_changes and not self._applying_internally:
            self.on_changes(delta, None)
        return delta

    def _apply_text_append(self, entity_id: str, forward: dict[str, Any]) -> None:
        """Apply a remote text_append op to the cached entity."""
        entity = self._entity_cache.get(entity_id)
        if entity is None:
            raise KeyError(f"_apply_text_append: entity {entity_id!r} not in store")
        chunk = forward.get("chunk", "")
        content: dict[str, Any] = getattr(entity, "content", None) or {}
        content["text"] = content.get("text", "") + chunk
        if getattr(entity, "content", None) is not content:
            entity.content = content  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Custom-op dispatch in apply_delta
    # ------------------------------------------------------------------

    def apply_delta(self, delta: Delta, origin: str | None = None) -> Delta:
        """Override to intercept custom-op keys before standard processing."""
        # Separate custom ops from regular changes.
        regular_data: DeltaData = {}
        custom_ops: list[tuple[str, str, Change]] = []  # (entity_id, op_name, change)

        for key, change_data in delta.asdict().items():
            if OP_SEP in key:
                entity_id, op_name = key.split(OP_SEP, 1)
                eid, reverse, forward = change_data
                custom_ops.append((entity_id, op_name, Change(eid, reverse, forward)))
            else:
                regular_data[key] = change_data

        # Apply custom ops
        for entity_id, op_name, change in custom_ops:
            handler = getattr(self, f"_apply_{op_name}", None)
            if handler is None:
                log.warning("Unknown custom op %r for entity %s", op_name, entity_id)
                continue
            self._applying_internally = True
            try:
                handler(entity_id, change.forward_component)
            finally:
                self._applying_internally = False

        # Fire on_changes for custom ops only (custom-op-only delta)
        if custom_ops and self.on_changes:
            custom_data: DeltaData = {}
            for entity_id, op_name, change in custom_ops:
                op_key = f"{entity_id}{OP_SEP}{op_name}"
                custom_data[op_key] = list(change.asdict())
            self.on_changes(Delta.from_data(custom_data), origin)

        # Delegate regular changes to the base class
        if regular_data:
            regular_delta = Delta.from_data(regular_data)
            return super().apply_delta(regular_delta, origin)

        return delta
