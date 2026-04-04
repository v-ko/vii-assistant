from __future__ import annotations

import base64
from typing import Optional

from PySide6.QtCore import QEvent, QPoint, QSize, Qt, Signal
from PySide6.QtGui import QCursor, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from assistant.inference.context import ContentType
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
        self._label.setStyleSheet("border: 1px solid palette(mid);")
        layout.addWidget(self._label)

    def show_pixmap(self, pixmap: QPixmap, anchor: QPoint) -> None:
        screen = QApplication.screenAt(anchor)
        if screen is None:
            screen = QApplication.primaryScreen()
        screen_rect = screen.availableGeometry()
        max_width = int(screen_rect.width() * 0.9)
        max_height = int(screen_rect.height() * 0.9)

        if pixmap.width() > max_width or pixmap.height() > max_height:
            target = pixmap.scaled(
                max_width,
                max_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        else:
            target = pixmap

        self._label.setPixmap(target)
        self.resize(self.sizeHint())
        # Keep the popup within screen bounds
        x = min(anchor.x(), screen_rect.right() - self.width())
        y = min(anchor.y(), screen_rect.bottom() - self.height())
        self.move(QPoint(max(x, screen_rect.x()), max(y, screen_rect.y())))
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
    size_hint_changed = Signal()

    def __init__(self, state: ContextItemViewState):
        super().__init__()
        self._state = state
        # Frameless & minimal
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet("QFrame { background: transparent; }")
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

    def _notify_size_change(self) -> None:
        self._layout.activate()
        self.updateGeometry()
        self.size_hint_changed.emit()


class _TextItemWidget(_BaseItemWidget):
    def __init__(self, state: ContextItemViewState):
        super().__init__(state)
        self._body = QLabel()
        self._body.setWordWrap(True)
        self._body.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self._body.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._body.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
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
                self._body.setStyleSheet(
                    "QLabel { color: palette(disabled-text); font-style: italic; }"
                )
            else:
                self._body.setText("")
                self._body.setStyleSheet("QLabel { color: palette(disabled-text); }")
        else:
            self._body.setText(trimmed)
            self._body.setStyleSheet("")
        self._notify_size_change()


class _ToolCallItemWidget(_BaseItemWidget):
    def __init__(self, state: ContextItemViewState):
        super().__init__(state)
        self._body = QLabel()
        self._body.setWordWrap(True)
        self._body.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self._body.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._body.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        self._insert_before_footer(self._body)
        self._state.tool_call_changed.connect(self._update_tool_call)
        self._update_tool_call(self._state.tool_call)

    def _update_tool_call(self, payload: str) -> None:
        if not payload.strip():
            self._body.setText("(…)")
            self._body.setStyleSheet(
                "QLabel { font-style: italic; font-family: monospace; }"
            )
        else:
            self._body.setText(payload)
            self._body.setStyleSheet("QLabel { font-family: monospace; }")
        self._notify_size_change()


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
        self._thumb.setStyleSheet("QLabel { border: 1px solid palette(mid); }")
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
            self._notify_size_change()
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
        self._notify_size_change()

    def enterEvent(self, event):
        cursor_pos = QCursor.pos()
        self._preview_manager.show(self._full_pixmap, cursor_pos + QPoint(20, 12))
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._preview_manager.hide()
        super().leaveEvent(event)


class _SystemPromptItemWidget(_BaseItemWidget):
    """Collapsible widget (expander/accordion) for displaying system prompt."""

    def __init__(self, state: ContextItemViewState):
        super().__init__(state)

        # Clickable header with toggle button and label
        self._header = QWidget()
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        self._header.mousePressEvent = lambda event: self._toggle_collapsed()
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(2, 2, 2, 2)
        header_layout.setSpacing(4)

        self._toggle_btn = QToolButton()
        self._toggle_btn.setArrowType(Qt.ArrowType.RightArrow)
        self._toggle_btn.setStyleSheet(
            "QToolButton { border: none; background: transparent; }"
        )
        self._toggle_btn.setFixedSize(16, 16)
        header_layout.addWidget(self._toggle_btn)

        title_label = QLabel("System Prompt")
        title_label.setStyleSheet(
            "QLabel { color: palette(disabled-text); font-weight: bold; font-size:"
            " 11px; }"
        )
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        self._layout.addWidget(self._header)

        # Collapsible content
        self._content = QLabel()
        self._content.setWordWrap(True)
        self._content.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self._content.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._content.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        self._content.setStyleSheet(
            "QLabel { color: palette(text); padding: 6px; "
            "border-left: 2px solid palette(mid); font-size: 11px; }"
        )
        self._content.setVisible(False)  # Start collapsed
        self._content.setMaximumHeight(0)  # Ensure it takes no space when hidden
        self._layout.addWidget(self._content)

        self._state.text_changed.connect(self._update_text)
        self._update_text(self._state.text)

    def _toggle_collapsed(self):
        is_visible = self._content.isVisible()
        new_visible = not is_visible

        self._content.setVisible(new_visible)
        # Set max height to control space usage
        if new_visible:
            self._content.setMaximumHeight(16777215)  # QWIDGETSIZE_MAX
        else:
            self._content.setMaximumHeight(0)

        self._toggle_btn.setArrowType(
            Qt.ArrowType.DownArrow if new_visible else Qt.ArrowType.RightArrow
        )
        self._notify_size_change()

    def _update_text(self, text: str) -> None:
        self._content.setText(text.strip() if text else "(no system prompt)")
        if self._content.isVisible():
            self._notify_size_change()


class ContextViewerWidget(QWidget):
    message_submitted = Signal(str)

    def __init__(self, state: ContextViewerState, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._state = state
        self._preview_manager = _ImagePreviewManager()
        self._request_in_progress = False
        self._interactions_enabled = state.interactions_enabled
        self._item_widgets: dict[str, tuple[QListWidgetItem, _BaseItemWidget]] = {}
        self._setup_ui()
        self._state.items_changed.connect(self._rebuild)
        self._state.interactions_enabled_changed.connect(
            self._apply_interactions_enabled
        )
        self._apply_interactions_enabled(self._interactions_enabled)
        self._rebuild()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._empty_label = QLabel("No context items yet.")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color: palette(disabled-text);")

        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._list.setFrameShape(QFrame.Shape.NoFrame)
        self._list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
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
        self._list.viewport().installEventFilter(self)

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
        self._item_widgets.clear()
        for state in items:
            widget = self._widget_for_state(state)
            list_item = QListWidgetItem()
            list_item.setSizeHint(widget.sizeHint())
            self._list.addItem(list_item)
            self._list.setItemWidget(list_item, widget)
            self._register_item_widget(state.item_id, list_item, widget)
        self._list.setUpdatesEnabled(True)
        self._update_all_item_sizes()
        self._list.scrollToBottom()

    def _widget_for_state(self, state: ContextItemViewState) -> _BaseItemWidget:
        kind = state.content_kind
        origin = state.origin
        if origin == "system":
            return _SystemPromptItemWidget(state)
        if kind == ContentType.IMAGE.value:
            return _ImageItemWidget(state, self._preview_manager)
        if kind == ContentType.TOOL_CALL.value:
            return _ToolCallItemWidget(state)
        return _TextItemWidget(state)

    def _register_item_widget(
        self, item_id: str, list_item: QListWidgetItem, widget: _BaseItemWidget
    ) -> None:
        self._item_widgets[item_id] = (list_item, widget)
        widget.size_hint_changed.connect(
            lambda item_id=item_id: self._update_item_size(item_id)
        )
        self._update_item_size(item_id)

    def _is_scrolled_near_bottom(self) -> bool:
        sb = self._list.verticalScrollBar()
        if sb is None:
            return True
        return sb.value() >= sb.maximum() - 30

    def _update_item_size(self, item_id: str) -> None:
        entry = self._item_widgets.get(item_id)
        if not entry:
            return
        list_item, widget = entry
        width = self._available_item_width()
        if width <= 0:
            return
        was_near_bottom = self._is_scrolled_near_bottom()
        widget.setFixedWidth(width)
        hint_height = widget.sizeHint().height()
        list_item.setSizeHint(QSize(width, hint_height))
        self._list.doItemsLayout()
        if was_near_bottom:
            self._list.scrollToBottom()

    def _available_item_width(self) -> int:
        viewport = self._list.viewport()
        if viewport is None:
            return 0
        # subtract a small padding to account for frame/margins
        return max(viewport.width() - 12, 0)

    def _update_all_item_sizes(self) -> None:
        for item_id in list(self._item_widgets.keys()):
            self._update_item_size(item_id)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_all_item_sizes()

    def eventFilter(self, obj, event):
        if obj is self._list.viewport() and event.type() == QEvent.Type.Resize:
            self._update_all_item_sizes()
        return super().eventFilter(obj, event)

    def _sync_send_enabled(self, _: str) -> None:
        self._refresh_composer_controls()

    def _on_submit_clicked(self) -> None:
        if self._request_in_progress or not self._interactions_enabled:
            return
        text = self._input_edit.text()
        self.message_submitted.emit(text)
        self._input_edit.clear()
        self._sync_send_enabled(self._input_edit.text())

    def set_request_in_progress(self, value: bool) -> None:
        if self._request_in_progress == value:
            return
        self._request_in_progress = value
        self._refresh_composer_controls()

    def _apply_interactions_enabled(self, enabled: bool) -> None:
        if self._interactions_enabled != enabled:
            self._interactions_enabled = enabled
        self._refresh_composer_controls()

    def _refresh_composer_controls(self) -> None:
        allow_input = self._interactions_enabled
        self._input_edit.setEnabled(allow_input)
        can_send = self._interactions_enabled and not self._request_in_progress
        self._send_button.setEnabled(can_send)
