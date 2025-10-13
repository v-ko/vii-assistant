from typing import TYPE_CHECKING, Optional
from uuid import uuid4

from PySide6.QtCore import (
    QCoreApplication,
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    Qt,
)
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import QHBoxLayout, QWidget

from assistant.inference.context import ContextItem
from assistant.util import pixmap_to_base64
from assistant.utils.capture_utils import clipboard_image, take_screenshot
from assistant.widgets.context_viewer import ContextViewerWidget
from assistant.widgets.settings import SettingsWidget

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from assistant.facade import Facade
    from assistant.view_states.terminal import TerminalViewState


class TerminalWindow(QWidget):
    def __init__(self, state: "TerminalViewState", parent=None):
        super().__init__(
            parent,
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint,
        )
        self._state = state
        self._facade: Optional["Facade"] = None
        # Make the window background transparent
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setup_ui()
        self.position_window()
        self.setup_shortcuts()
        # self.setup_animations()
        self._bind_state()

    def setup_ui(self):
        # Create a container widget to hold the UI; this widget will be
        # animated.
        self.container = QWidget(self)
        container_layout = QHBoxLayout(self.container)
        container_layout.setContentsMargins(10, 10, 10, 10)

        # Left side: context viewer widget
        self.context_viewer = ContextViewerWidget(self._state.context_view)
        container_layout.addWidget(self.context_viewer, 1)

        # Model settings widget
        self.model_settings = SettingsWidget(self._state.settings)
        container_layout.addWidget(self.model_settings, 1)
        container_layout.setStretch(0, 1)
        container_layout.setStretch(1, 1)

        # Apply styling to the container only (the window is transparent)
        self.container.setStyleSheet(
            """
            QWidget {
                background-color: #2d2d2d;
                border: 1px solid #444;
                border-radius: 5px;
                color: #e0e0e0;
            }
        """
        )

    def set_facade(self, facade: "Facade") -> None:
        self._facade = facade

    def position_window(self):
        """
        Position the window so that its top edge is at 20px.
        The container fills the window and will be animated.
        """
        screen = QGuiApplication.primaryScreen()
        geometry = screen.availableGeometry()
        window_width = int(geometry.width() * 0.9)
        window_height = geometry.height() // 2
        window_x = geometry.x() + (geometry.width() - window_width) // 2
        window_y = 20  # Final top position

        self.setGeometry(window_x, window_y, window_width, window_height)
        self.container.setGeometry(0, 0, window_width, window_height)

    def showEvent(self, event):
        super().showEvent(event)
        # Start with the container above the visible area (within the window).
        self.container.move(0, -self.container.height())
        show_animation = QPropertyAnimation(self.container, b"pos", self)
        show_animation.setDuration(300)
        show_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        show_animation.setStartValue(self.container.pos())
        show_animation.setEndValue(QPoint(0, 0))
        show_animation.start()

    def hide(self):
        # Prevent immediate hiding; animate slide-up instead.
        # event.ignore()
        print("At hideEvent")
        hide_animation = QPropertyAnimation(self.container, b"pos", self)
        hide_animation.setDuration(300)
        hide_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        hide_animation.setStartValue(self.container.pos())
        hide_animation.setEndValue(QPoint(0, -self.container.height()))
        hide_animation.finished.connect(lambda: QWidget.hide(self))
        hide_animation.start()

    def _bind_state(self):
        settings_state = self._state.settings
        settings_state.request_in_progress_changed.connect(
            self.context_viewer.set_request_in_progress
        )
        self.context_viewer.set_request_in_progress(settings_state.request_in_progress)

    def attach_screen(self) -> None:
        if not self._facade:
            return
        screen = self._facade.watched_screen or self._facade.get_default_screen()
        pixmap = take_screenshot(screen)
        if pixmap is None:
            print("Failed to capture screenshot for attachment.")
            return
        self._add_image_item(pixmap, "screenshot")

    def attach_clipboard(self) -> None:
        if not self._facade:
            return
        pixmap = clipboard_image()
        if pixmap is None:
            print("No image found in clipboard.")
            return
        self._add_image_item(pixmap, "clipboard")

    def _add_image_item(self, pixmap, source: str) -> None:
        if not self._facade:
            return
        encoded = pixmap_to_base64(pixmap)
        if not encoded:
            print("Failed to encode captured image.")
            return
        manager = self._facade.context_manager
        position = manager.next_position()
        item = ContextItem(
            id=uuid4().hex,
            position=position,
            size=(pixmap.width(), pixmap.height()),
            content={"image": encoded},
            request=None,
            metadata={"origin": "user", "source": source},
        )
        self._facade.add_context_item(item)

    def setup_shortcuts(self):
        """Set up keyboard shortcuts."""
        self.alt_f4_shortcut = QShortcut(QKeySequence("Ctrl+w"), self)
        self.alt_f4_shortcut.activated.connect(self.quit_application)

    def quit_application(self):
        """Quit the application."""
        QCoreApplication.quit()

    def closeEvent(self, event):
        self.hide()
