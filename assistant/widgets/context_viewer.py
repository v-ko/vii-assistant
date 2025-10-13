from __future__ import annotations

import base64
from typing import Optional

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QCursor, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from assistant.inference.context import ContentKind
from assistant.view_states.context_view import ContextItemViewState, ContextViewerState


def _decode_pixmap(b64: str) -> Optional[QPixmap]:
    if not b64:
        return None
    try:
        data = base64.b64decode(b64)
    except Exception:
        return None
    image = QImage.fromData(data)
    if image.isNull():
        return None
    return QPixmap.fromImage(image)


def _short_text(text: str, limit: int = 280) -> str:
    clean = text.strip()
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "…"


class _ImagePreviewPopup(QWidget):
    def __init__(self):
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setWindowOpacity(0.98)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(0)
        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet(
            "background-color: rgba(20, 20, 20, 220); border: 1px solid #555;"
        )
        layout.addWidget(self._label)

    def show_pixmap(self, pixmap: QPixmap, anchor: QPoint) -> None:
        target = pixmap
        max_width = 480
        max_height = 360
        if (
            pixmap.width() > max_width
            or pixmap.height() > max_height
            and pixmap.width() > 0
            and pixmap.height() > 0
        ):
            target = pixmap.scaled(
                max_width,
                max_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self._label.setPixmap(target)
        self.resize(self.sizeHint())
        self.move(anchor)
        self.show()


class _ImagePreviewManager:
    def __init__(self):
        self._popup: Optional[_ImagePreviewPopup] = None

    def show(self, pixmap: Optional[QPixmap], anchor: QPoint) -> None:
        if pixmap is None:
            return
        if self._popup is None:
            self._popup = _ImagePreviewPopup()
        self._popup.show_pixmap(pixmap, anchor)

    def hide(self) -> None:
        if self._popup:
            self._popup.hide()


class _BaseItemWidget(QFrame):
    def __init__(self, state: ContextItemViewState):
        super().__init__()
        self._state = state
        # Frameless & minimal
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet("QFrame { background-color: transparent; }")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 2, 4, 2)
        self._layout.setSpacing(2)

        # Use tooltip for request metadata (not always visible)
        self._state.request_summary_changed.connect(self._update_tooltip)
        self._state.origin_changed.connect(self._update_tooltip)
        self._update_tooltip(self._state.request_summary)

    def _insert_before_footer(self, widget: QWidget) -> None:
        # With minimal layout just add widget
        self._layout.addWidget(widget)

    def _update_tooltip(self, _summary: str) -> None:  # summary unused directly
        origin = self._state.origin
        summary = self._state.request_summary
        parts = []
        if origin:
            parts.append(origin)
        if summary:
            parts.append(summary)
        self.setToolTip(" \u2022 ".join(parts) if parts else "")


class _TextItemWidget(_BaseItemWidget):
    def __init__(self, state: ContextItemViewState):
        super().__init__(state)
        self._body = QLabel()
        self._body.setWordWrap(True)
        self._body.setMinimumWidth(220)
        self._insert_before_footer(self._body)
        self._state.text_changed.connect(self._update_text)
        self._state.request_summary_changed.connect(
            lambda _: self._update_text(self._state.text)
        )
        self._update_text(self._state.text)

    def _update_text(self, text: str) -> None:
        trimmed = text.strip()
        if not trimmed:
            # Show placeholder if request in progress (has request summary) else nothing
            if self._state.request_summary:
                self._body.setText("(…)")
                self._body.setStyleSheet("QLabel { color: #888; font-style: italic; }")
            else:
                self._body.setText("")
                self._body.setStyleSheet("QLabel { color: #888; }")
        else:
            self._body.setText(_short_text(trimmed))
            self._body.setStyleSheet("QLabel { color: #f0f0f0; }")


class _ToolCallItemWidget(_BaseItemWidget):
    def __init__(self, state: ContextItemViewState):
        super().__init__(state)
        self._body = QLabel()
        self._body.setWordWrap(True)
        self._insert_before_footer(self._body)
        self._state.tool_call_changed.connect(self._update_tool_call)
        self._update_tool_call(self._state.tool_call)

    def _update_tool_call(self, payload: str) -> None:
        if not payload.strip():
            self._body.setText("(…)")
            self._body.setStyleSheet(
                "QLabel { color: #bfa86a; font-style: italic; font-family: monospace; }"
            )
        else:
            self._body.setText(_short_text(payload, limit=200))
            self._body.setStyleSheet(
                "QLabel { color: #ffd27f; font-family: monospace; }"
            )


class _ImageItemWidget(_BaseItemWidget):
    def __init__(
        self,
        state: ContextItemViewState,
        preview_manager: _ImagePreviewManager,
    ):
        super().__init__(state)
        self._preview_manager = preview_manager
        self._thumb = QLabel()
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb.setStyleSheet(
            "QLabel { background-color: #1f1f1f; border: 1px solid #333; }"
        )
        self._insert_before_footer(self._thumb)
        self._full_pixmap: Optional[QPixmap] = None
        self._state.image_b64_changed.connect(self._update_image)
        self._update_image(self._state.image_b64)

    def _update_image(self, b64: str) -> None:
        pixmap = _decode_pixmap(b64)
        self._full_pixmap = pixmap
        if pixmap is None:
            self._thumb.setPixmap(QPixmap())
            self._thumb.setText("(invalid image)")
            self._thumb.setMinimumSize(128, 96)
            return
        scaled = pixmap.scaled(
            128,
            96,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._thumb.setText("")
        self._thumb.setPixmap(scaled)
        self._thumb.setMinimumSize(scaled.size())

    def enterEvent(self, event):
        cursor_pos = QCursor.pos()
        self._preview_manager.show(self._full_pixmap, cursor_pos + QPoint(20, 12))
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._preview_manager.hide()
        super().leaveEvent(event)


class ContextViewerWidget(QWidget):
    message_submitted = Signal(str)

    def __init__(self, state: ContextViewerState, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._state = state
        self._preview_manager = _ImagePreviewManager()
        self._request_in_progress = False
        self._setup_ui()
        self._state.items_changed.connect(self._rebuild)
        self._rebuild()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._empty_label = QLabel("No context items yet.")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color: #777;")

        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._list.setFrameShape(QFrame.Shape.NoFrame)
        self._list.setSpacing(2)
        self._list.setStyleSheet(
            """
            QListWidget {
                background-color: transparent;
            }
            QListWidget::item {
                margin: 0px;
            }
            """
        )

        layout.addWidget(self._empty_label)
        layout.addWidget(self._list)

        self._composer_row = QHBoxLayout()
        self._composer_row.setContentsMargins(6, 0, 6, 6)
        self._composer_row.setSpacing(6)

        self._input_edit = QLineEdit()
        self._input_edit.setPlaceholderText("Type a message…")
        self._input_edit.setClearButtonEnabled(True)
        self._input_edit.returnPressed.connect(self._on_submit_clicked)
        self._input_edit.textChanged.connect(self._sync_send_enabled)
        self._composer_row.addWidget(self._input_edit, 1)

        self._send_button = QPushButton("Send")
        self._send_button.setDefault(True)
        self._send_button.clicked.connect(self._on_submit_clicked)
        self._send_button.setEnabled(True)
        self._composer_row.addWidget(self._send_button)

        layout.addLayout(self._composer_row)

    def _rebuild(self) -> None:
        items = self._state.sorted_items()
        if not items:
            self._list.hide()
            self._empty_label.show()
            return

        self._empty_label.hide()
        self._list.show()
        self._list.setUpdatesEnabled(False)
        self._list.clear()
        for state in items:
            widget = self._widget_for_state(state)
            list_item = QListWidgetItem()
            list_item.setSizeHint(widget.sizeHint())
            self._list.addItem(list_item)
            self._list.setItemWidget(list_item, widget)
        self._list.setUpdatesEnabled(True)

    def _widget_for_state(self, state: ContextItemViewState) -> QWidget:
        kind = state.content_kind
        if kind == ContentKind.IMAGE.value:
            return _ImageItemWidget(state, self._preview_manager)
        if kind == ContentKind.TOOL_CALL.value:
            return _ToolCallItemWidget(state)
        return _TextItemWidget(state)

    def _sync_send_enabled(self, _: str) -> None:
        self._send_button.setEnabled(not self._request_in_progress)

    def _on_submit_clicked(self) -> None:
        if self._request_in_progress:
            return
        text = self._input_edit.text()
        self.message_submitted.emit(text)
        self._input_edit.clear()
        self._sync_send_enabled(self._input_edit.text())

    def set_request_in_progress(self, value: bool) -> None:
        if self._request_in_progress == value:
            return
        self._request_in_progress = value
        self._sync_send_enabled(self._input_edit.text())
