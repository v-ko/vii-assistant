"""QAbstractListModel wrapping ContextViewerState for QML."""

from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum

from PySide6.QtCore import (
    QAbstractListModel,
    QByteArray,
    QModelIndex,
    QObject,
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


class ContextListModel(QAbstractListModel):
    """Exposes ContextViewerState items as a flat list model for QML."""

    def __init__(self, state: ContextViewerState, parent=None):
        super().__init__(parent)
        self._state = state
        self._items: list[ContextItemViewState] = []
        self._connections: dict[str, list] = {}
        state.items_changed.connect(self._rebuild)
        self._rebuild()

    def _rebuild(self):
        # Disconnect old item signals
        for item_id, conns in self._connections.items():
            for conn in conns:
                try:
                    QObject.disconnect(conn)
                except RuntimeError:
                    pass
        self._connections.clear()

        self.beginResetModel()
        self._items = self._state.sorted_items()
        self.endResetModel()

        # Connect per-item change signals to dataChanged
        for i, item in enumerate(self._items):
            conns = []

            def _make_notifier(row, roles):
                def _notify(*_args):
                    model_idx = self.index(row, 0)
                    self.dataChanged.emit(model_idx, model_idx, roles)

                return _notify

            conns.append(
                item.text_changed.connect(_make_notifier(i, [ContextItemRole.Text]))
            )
            conns.append(
                item.image_b64_changed.connect(
                    _make_notifier(i, [ContextItemRole.ImageB64])
                )
            )
            conns.append(
                item.request_summary_changed.connect(
                    _make_notifier(i, [ContextItemRole.RequestSummary])
                )
            )
            conns.append(
                item.origin_changed.connect(_make_notifier(i, [ContextItemRole.Origin]))
            )
            conns.append(
                item.content_kind_changed.connect(
                    _make_notifier(i, [ContextItemRole.ContentKind])
                )
            )
            self._connections[item.item_id] = conns

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
        }
