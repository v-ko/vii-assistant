from __future__ import annotations

from collections.abc import Iterable

from fusion.libs.entity.change import Change
from fusion.platform.qt_widgets import Property
from PySide6.QtCore import QObject, Signal

from assistant.inference.context import ContentType, ContextItem


def _summarize_request(item: ContextItem) -> str:
    payload = item.request or {}
    if not isinstance(payload, dict):
        return ""
    parts: list[str] = []
    status = payload.get("result")
    if status:
        parts.append(f"result={status}")
    temperature = payload.get("temperature")
    if temperature is not None:
        parts.append(f"temp={temperature}")
    max_tokens = payload.get("max_new_tokens")
    if max_tokens is not None:
        parts.append(f"max_tokens={max_tokens}")
    if not parts:
        return ""
    return "Request: " + ", ".join(parts)


class ContextItemViewState(QObject):
    position_changed = Signal(int)
    content_kind_changed = Signal(str)
    text_changed = Signal(str)
    image_b64_changed = Signal(str)
    tool_call_changed = Signal(str)
    request_summary_changed = Signal(str)
    origin_changed = Signal(str)

    def __init__(self, item_id: str, parent: QObject | None = None):
        super().__init__(parent)
        self._item_id = item_id
        self._position = 0
        self._content_kind = ContentType.TEXT.value
        self._text = ""
        self._image_b64 = ""
        self._tool_call = ""
        self._request_summary = ""
        self._origin = ""

    @Property(str, constant=True)
    def item_id(self) -> str:
        return self._item_id

    @Property(int, notify=position_changed)
    def position(self) -> int:
        return self._position

    @position.setter
    def position(self, value: int) -> None:
        if self._position == value:
            return
        self._position = value
        self.position_changed.emit(value)

    @Property(str, notify=content_kind_changed)
    def content_kind(self) -> str:
        return self._content_kind

    @content_kind.setter
    def content_kind(self, value: str) -> None:
        if self._content_kind == value:
            return
        self._content_kind = value
        self.content_kind_changed.emit(value)

    @Property(str, notify=text_changed)
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, value: str) -> None:
        if self._text == value:
            return
        self._text = value
        self.text_changed.emit(value)

    @Property(str, notify=image_b64_changed)
    def image_b64(self) -> str:
        return self._image_b64

    @image_b64.setter
    def image_b64(self, value: str) -> None:
        if self._image_b64 == value:
            return
        self._image_b64 = value
        self.image_b64_changed.emit(value)

    @Property(str, notify=tool_call_changed)
    def tool_call(self) -> str:
        return self._tool_call

    @tool_call.setter
    def tool_call(self, value: str) -> None:
        if self._tool_call == value:
            return
        self._tool_call = value
        self.tool_call_changed.emit(value)

    @Property(str, notify=request_summary_changed)
    def request_summary(self) -> str:
        return self._request_summary

    @request_summary.setter
    def request_summary(self, value: str) -> None:
        if self._request_summary == value:
            return
        self._request_summary = value
        self.request_summary_changed.emit(value)

    @Property(str, notify=origin_changed)
    def origin(self) -> str:
        return self._origin

    @origin.setter
    def origin(self, value: str) -> None:
        if self._origin == value:
            return
        self._origin = value
        self.origin_changed.emit(value)

    def apply_context_item(self, item: ContextItem) -> bool:
        kind = item.content_type().value
        reposition = item.position != self._position
        previous_kind = self._content_kind
        origin = (item.metadata or {}).get("origin", "") if item.metadata else ""

        self.position = item.position
        self.content_kind = kind
        self.origin = origin
        self.request_summary = _summarize_request(item)

        if kind == ContentType.TEXT.value:
            text = str(item.content.get("text", ""))
            self.text = text
            self.image_b64 = ""
            self.tool_call = ""
        elif kind == ContentType.IMAGE.value:
            image_b64 = item.content.get("image")
            self.image_b64 = image_b64 if isinstance(image_b64, str) else ""
            self.text = ""
            self.tool_call = ""
        else:
            payload = item.content.get("tool_call")
            self.tool_call = str(payload) if payload is not None else ""
            self.text = ""
            self.image_b64 = ""

        kind_changed = previous_kind != kind
        return reposition or kind_changed


class ContextViewerState(QObject):
    items_changed = Signal()
    interactions_enabled_changed = Signal(bool)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._items: dict[str, ContextItemViewState] = {}
        self._interactions_enabled = False

    @Property(list, notify=items_changed)
    def items(self) -> list[ContextItemViewState]:
        return self.sorted_items()

    def sorted_items(self) -> list[ContextItemViewState]:
        return sorted(
            self._items.values(),
            key=lambda state: (state.position, state.item_id),
        )

    @Property(bool, notify=interactions_enabled_changed)
    def interactions_enabled(self) -> bool:
        return self._interactions_enabled

    @interactions_enabled.setter
    def interactions_enabled(self, value: bool) -> None:
        if self._interactions_enabled == value:
            return
        self._interactions_enabled = value
        self.interactions_enabled_changed.emit(value)

    def get_item(self, item_id: str) -> ContextItemViewState | None:
        return self._items.get(item_id)

    def apply_change(self, change: Change) -> None:
        # Deletion
        if change.is_delete():
            if isinstance(change.old_state, ContextItem):
                key: str = str(change.old_state.id)
                state = self._items.pop(key, None)
                if state is not None:
                    state.setParent(None)
                    state.deleteLater()
                    self.items_changed.emit()
            return

        # Creation / update
        item = change.new_state
        if not isinstance(item, ContextItem):  # Ignore irrelevant changes
            return
        key: str = str(item.id)
        state = self._items.get(key)
        created = False
        if state is None:
            state = ContextItemViewState(key, parent=self)
            self._items[key] = state
            created = True
        refresh_needed = state.apply_context_item(item)
        if created or refresh_needed:
            self.items_changed.emit()

    def apply_changes(self, changes: Iterable[Change]) -> None:
        for change in changes:
            self.apply_change(change)

    def replace_all(self, items: Iterable[ContextItem]) -> None:
        current_ids = set(self._items.keys())
        next_ids: set[str] = set()
        needs_emit = False
        for item in items:
            if not isinstance(item, ContextItem):  # defensive
                continue
            key: str = str(item.id)
            next_ids.add(key)
            state = self._items.get(key)
            if state is None:
                state = ContextItemViewState(key, parent=self)
                self._items[key] = state
                needs_emit = True
            if state.apply_context_item(item):
                needs_emit = True
        removed = current_ids - next_ids
        for removed_id in removed:
            state = self._items.pop(removed_id, None)
            if state is not None:
                state.setParent(None)
                state.deleteLater()
                needs_emit = True
        if needs_emit:
            self.items_changed.emit()
