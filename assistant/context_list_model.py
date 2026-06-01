"""QAbstractListModel wrapping ContextViewerState for QML."""

from __future__ import annotations

import bisect
from enum import IntEnum

from PySide6.QtCore import (
    QAbstractListModel,
    QByteArray,
    QModelIndex,
    QPersistentModelIndex,
    Qt,
)

from assistant.view_states.context_view import ContextItemViewState, ContextViewerState


class ContextItemRole(IntEnum):
    ItemId = Qt.ItemDataRole.UserRole + 1
    Position = Qt.ItemDataRole.UserRole + 2
    ContentKind = Qt.ItemDataRole.UserRole + 3
    Text = Qt.ItemDataRole.UserRole + 4
    ImageB64 = Qt.ItemDataRole.UserRole + 5
    RequestSummary = Qt.ItemDataRole.UserRole + 6
    Origin = Qt.ItemDataRole.UserRole + 7
    DisplayText = Qt.ItemDataRole.UserRole + 8
    FocusMode = Qt.ItemDataRole.UserRole + 9


def _sort_key(item: ContextItemViewState):
    return (item.position, item.item_id)


class ContextListModel(QAbstractListModel):
    """Exposes ContextViewerState items as a flat list model for QML.

    Uses incremental insert/remove to preserve ListView scroll position.
    Property changes (streaming text, etc.) are delivered via a single
    item_updated signal from the view state — no per-item connections needed.
    """

    def __init__(self, state: ContextViewerState, parent=None):
        super().__init__(parent)
        self._state = state
        self._items: list[ContextItemViewState] = []

        state.item_added.connect(self._on_item_added)
        state.item_removed.connect(self._on_item_removed)
        state.item_moved.connect(self._on_item_moved)
        state.item_updated.connect(self._on_item_updated)
        state.items_changed.connect(self._full_rebuild)
        self._full_rebuild()

    # ── Incremental operations ──────────────────────────────────

    def _on_item_added(self, item_id: str) -> None:
        item = self._state.get_item(item_id)
        if item is None:
            return
        key = _sort_key(item)
        row = bisect.bisect_left([_sort_key(it) for it in self._items], key)
        self.beginInsertRows(QModelIndex(), row, row)
        self._items.insert(row, item)
        self.endInsertRows()

    def _on_item_removed(self, item_id: str) -> None:
        row = self._find_row(item_id)
        if row is None:
            return
        self.beginRemoveRows(QModelIndex(), row, row)
        self._items.pop(row)
        self.endRemoveRows()

    def _on_item_moved(self, item_id: str) -> None:
        old_row = self._find_row(item_id)
        if old_row is None:
            return
        item = self._items[old_row]
        key = _sort_key(item)
        temp = self._items[:old_row] + self._items[old_row + 1 :]
        new_row = bisect.bisect_left([_sort_key(it) for it in temp], key)
        if new_row == old_row:
            return
        dest = new_row if new_row < old_row else new_row + 1
        self.beginMoveRows(QModelIndex(), old_row, old_row, QModelIndex(), dest)
        self._items.pop(old_row)
        self._items.insert(new_row, item)
        self.endMoveRows()

    def _on_item_updated(self, item_id: str) -> None:
        """An existing item's properties changed (e.g. streaming text)."""
        row = self._find_row(item_id)
        if row is None:
            return
        idx = self.index(row, 0)
        self.dataChanged.emit(idx, idx, [])

    # ── Full rebuild (only for replace_all / session switch) ────

    def _full_rebuild(self):
        self.beginResetModel()
        self._items = self._state.sorted_items()
        self.endResetModel()

    # ── Helpers ─────────────────────────────────────────────────

    def _find_row(self, item_id: str) -> int | None:
        for i, item in enumerate(self._items):
            if item.item_id == item_id:
                return i
        return None

    def rowCount(
        self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()
    ) -> int:
        return len(self._items)

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> object:
        if not index.isValid() or index.row() >= len(self._items):
            return None
        item = self._items[index.row()]
        if role == ContextItemRole.ItemId:
            return item.item_id
        if role == ContextItemRole.Position:
            return item.position
        if role == ContextItemRole.ContentKind:
            return item.content_kind
        if role == ContextItemRole.Text:
            return item.text
        if role == ContextItemRole.ImageB64:
            return item.image_b64
        if role == ContextItemRole.RequestSummary:
            return item.request_summary
        if role == ContextItemRole.Origin:
            return item.origin
        if role == ContextItemRole.DisplayText:
            return item.display_text
        if role == ContextItemRole.FocusMode:
            return item.focus_mode
        return None

    def roleNames(self) -> dict[int, QByteArray]:
        return {
            int(ContextItemRole.ItemId): QByteArray(b"itemId"),
            int(ContextItemRole.Position): QByteArray(b"position"),
            int(ContextItemRole.ContentKind): QByteArray(b"contentKind"),
            int(ContextItemRole.Text): QByteArray(b"text"),
            int(ContextItemRole.ImageB64): QByteArray(b"imageB64"),
            int(ContextItemRole.RequestSummary): QByteArray(b"requestSummary"),
            int(ContextItemRole.Origin): QByteArray(b"origin"),
            int(ContextItemRole.DisplayText): QByteArray(b"displayText"),
            int(ContextItemRole.FocusMode): QByteArray(b"focusMode"),
        }
