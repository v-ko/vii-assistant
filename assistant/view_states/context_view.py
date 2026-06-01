from __future__ import annotations

from collections.abc import Iterable

from fusion.platform.qt_widgets import Property
from PySide6.QtCore import QObject, Signal

from assistant.inference.context import ContextItem, ImageItem, TextItem


def _summarize_request(item: ContextItem) -> str:
    payload = item.request or {}
    if not isinstance(payload, dict):
        return ""
    generation_params = payload.get("generation_params")
    if not isinstance(generation_params, dict):
        generation_params = {}
    parts: list[str] = []
    status = payload.get("result")
    if status:
        parts.append(f"result={status}")
    temperature = generation_params.get("temperature")
    if temperature is not None:
        parts.append(f"temp={temperature}")
    max_tokens = generation_params.get("max_new_tokens")
    if max_tokens is not None:
        parts.append(f"max_tokens={max_tokens}")
    if not parts:
        return ""
    return "Request: " + ", ".join(parts)


def _format_display_text(item: ContextItem) -> str:
    """Format item text for UI display. Prettifies tool call arguments."""
    if not isinstance(item, TextItem):
        return ""
    text = item.text
    meta = item.metadata or {}
    request = item.request or {}

    # Client execution request items have raw JSON arguments — format them
    if isinstance(request, dict) and request.get("execution") == "client":
        arguments = meta.get("arguments", {})
        focus_mode = request.get("focus_mode", "")
        if focus_mode == "python":
            code = arguments.get("code", "")
            return f"▶ {code}" if code else text
        elif focus_mode == "click_at":
            return f"⊕ click ({arguments.get('x', '?')}, {arguments.get('y', '?')})"
        elif focus_mode == "scroll":
            steps = arguments.get("steps", 0)
            direction = "↑" if steps > 0 else "↓"
            return f"{direction} scroll {abs(steps)}"
        elif focus_mode == "move_pointer":
            return f"→ move ({arguments.get('x', '?')}, {arguments.get('y', '?')})"
        return text

    # Tool call in metadata (completed assistant generation that triggered a tool)
    tool_call = meta.get("tool_call")
    if tool_call and isinstance(tool_call, dict):
        name = tool_call.get("name", "")
        args = tool_call.get("arguments", {})
        suffix = ""
        if name == "localization":
            suffix = f": {args.get('instruction', '')}"
        elif name == "python":
            code = args.get("code", "")
            suffix = f": {code[:80]}{'…' if len(code) > 80 else ''}"
        elif name in ("click_at", "scroll"):
            suffix = f": {args}"
        # Show text before tool call (strip markers) + short tool call summary
        # Strip <tool_call>...</tool_call> from display text
        display = (
            text.split("<tool_call>")[0].strip() if "<tool_call>" in text else text
        )
        parts = []
        if display:
            parts.append(display)
        parts.append(f"⚡ {name}{suffix}")
        return "\n".join(parts)

    return text


class ContextItemViewState(QObject):
    position_changed = Signal(int)
    content_kind_changed = Signal(str)
    text_changed = Signal(str)
    display_text_changed = Signal(str)
    image_b64_changed = Signal(str)
    request_summary_changed = Signal(str)
    origin_changed = Signal(str)
    focus_mode_changed = Signal(str)

    def __init__(self, item_id: str, parent: QObject | None = None):
        super().__init__(parent)
        self._item_id = item_id
        self._position = 0
        self._content_kind = "text"
        self._text = ""
        self._display_text = ""
        self._image_b64 = ""
        self._request_summary = ""
        self._origin = ""
        self._focus_mode = ""

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

    @Property(str, notify=display_text_changed)
    def display_text(self) -> str:
        return self._display_text

    @display_text.setter
    def display_text(self, value: str) -> None:
        if self._display_text == value:
            return
        self._display_text = value
        self.display_text_changed.emit(value)

    @Property(str, notify=focus_mode_changed)
    def focus_mode(self) -> str:
        return self._focus_mode

    @focus_mode.setter
    def focus_mode(self, value: str) -> None:
        if self._focus_mode == value:
            return
        self._focus_mode = value
        self.focus_mode_changed.emit(value)

    def apply_context_item(self, item: ContextItem) -> bool:
        reposition = item.position != self._position
        previous_kind = self._content_kind

        self.position = item.position
        self.origin = item.origin
        self.request_summary = _summarize_request(item)
        self.focus_mode = (item.metadata or {}).get("focus_mode", "")
        self.display_text = _format_display_text(item)

        if isinstance(item, TextItem):
            self.content_kind = "text"
            self.text = item.text
            self.image_b64 = ""
        elif isinstance(item, ImageItem):
            self.content_kind = "image"
            self.image_b64 = item.image_b64
            self.text = ""
        else:
            self.content_kind = "unknown"
            self.text = ""
            self.image_b64 = ""

        kind_changed = previous_kind != self._content_kind
        return reposition or kind_changed


class ContextViewerState(QObject):
    items_changed = Signal()  # bulk reset (replace_all)
    item_added = Signal(str)  # item_id
    item_removed = Signal(str)  # item_id
    item_moved = Signal(str)  # item_id (position changed)
    item_updated = Signal(str)  # item_id (properties changed, no structural change)
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

    def apply_entity(self, item: ContextItem) -> None:
        """Create or update a view state entry from a ContextItem entity."""
        key: str = str(item.id)
        state = self._items.get(key)
        if state is None:
            state = ContextItemViewState(key, parent=self)
            self._items[key] = state
            state.apply_context_item(item)
            self.item_added.emit(key)
        else:
            reposition = state.apply_context_item(item)
            if reposition:
                self.item_moved.emit(key)
            else:
                self.item_updated.emit(key)

    def remove_entity(self, entity_id: str) -> None:
        """Remove a view state entry by entity id."""
        state = self._items.pop(entity_id, None)
        if state is not None:
            state.setParent(None)
            state.deleteLater()
            self.item_removed.emit(entity_id)

    def apply_changes(self, changes: Iterable) -> None:
        # Kept for compatibility but not used in the new flow
        pass

    def replace_all(self, items: Iterable[ContextItem]) -> None:
        current_ids = set(self._items.keys())
        next_ids: set[str] = set()
        for item in items:
            if not isinstance(item, ContextItem):  # defensive
                continue
            key: str = str(item.id)
            next_ids.add(key)
            state = self._items.get(key)
            if state is None:
                state = ContextItemViewState(key, parent=self)
                self._items[key] = state
            state.apply_context_item(item)
        removed = current_ids - next_ids
        for removed_id in removed:
            state = self._items.pop(removed_id, None)
            if state is not None:
                state.setParent(None)
                state.deleteLater()
        self.items_changed.emit()
