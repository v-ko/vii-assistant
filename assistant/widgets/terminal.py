from base64 import b64encode
from io import BytesIO
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import (
    QCoreApplication,
    QEasingCurve,
    QEvent,
    QPoint,
    QPropertyAnimation,
    Qt,
)
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import QHBoxLayout, QWidget

from assistant.actions import (
    _ensure_session,
    add_user_message,
    new_session,
    ocr_clipboard,
    open_sessions_folder,
)
from assistant.facade import vii  # module-level singleton
from assistant.image_ops import resize_to_target
from assistant.inference.context import ContextItem
from assistant.model_configs import MODEL_SPECS, get_resolution_for_model
from assistant.utils.capture_utils import clipboard_image, take_screenshot
from assistant.utils.image_utils import qpixmap_to_pil
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
        # direct access through imported singleton
        self._facade: Optional["Facade"] = vii
        # Make the window background transparent
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setup_ui()
        self.position_window()
        self.setup_shortcuts()
        # self.setup_animations()
        self._bind_state()
        # Wire view signals to actions using facade singleton
        self.model_settings.ocr_clipboard_clicked.connect(ocr_clipboard)
        self.model_settings.attach_screen_clicked.connect(self.attach_screen)
        self.model_settings.attach_clipboard_clicked.connect(self.attach_clipboard)
        self.model_settings.new_session_clicked.connect(new_session)
        self.model_settings.open_sessions_folder_clicked.connect(open_sessions_folder)
        self.context_viewer.message_submitted.connect(add_user_message)

    def setup_ui(self):
        # Create a container widget to hold the UI; this widget will be
        # animated.
        self.container = QWidget(self)
        self.container.setObjectName("terminalWindowContainer")
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

        # Container styling (window itself is transparent for drop-down effect)
        self._apply_container_style()

    def _apply_container_style(self) -> None:
        p = self.palette()
        bg = p.color(p.ColorRole.Window).name()
        border = p.color(p.ColorRole.Mid).name()
        fg = p.color(p.ColorRole.WindowText).name()
        self.container.setStyleSheet(f"""
            #terminalWindowContainer {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: 5px;
                color: {fg};
            }}
        """)

    def changeEvent(self, event):
        if event.type() == QEvent.Type.PaletteChange:
            self._apply_container_style()
        super().changeEvent(event)

    # set_facade removed; widgets access facade singleton directly

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
        try:
            screen = self._facade.current_watched_screen()
        except Exception as e:
            print(f"Cannot attach screen: {e}")
            return
        pixmap = take_screenshot(screen)
        if pixmap is None:
            print("Failed to capture screenshot")
            return
        self._add_image_item(pixmap, "screenshot")

    def attach_clipboard(self) -> None:
        if not self._facade:
            return
        pixmap = clipboard_image()
        if pixmap is None:
            print("No image in clipboard")
            return
        self._add_image_item(pixmap, "clipboard")

    def _add_image_item(self, pixmap, source: str) -> None:
        if not self._facade:
            return

        _ensure_session()
        controller = self._facade.context_controller
        pil = qpixmap_to_pil(pixmap)
        model_key = self._facade.app_state.settings_VS.selected_model
        spec = MODEL_SPECS.get(model_key, {})
        if not spec.get("vision"):
            log.warning(
                "Selected model %r is not a vision model — skipping image attachment",
                model_key,
            )
            return
        target_w, target_h = get_resolution_for_model(model_key)
        processed_img, meta = resize_to_target(pil, target_w, target_h)

        buf = BytesIO()
        processed_img.save(buf, format="PNG")
        encoded = b64encode(buf.getvalue()).decode("ascii")

        if not encoded:
            raise RuntimeError("Image encoding failed")

        item = ContextItem.create_image(
            position=controller.next_position(),
            image_b64=encoded,
            width=meta["width"],
            height=meta["height"],
            origin=source,
        )
        controller.create(item)

    def setup_shortcuts(self):
        """Set up keyboard shortcuts."""
        self.alt_f4_shortcut = QShortcut(QKeySequence("Ctrl+w"), self)
        self.alt_f4_shortcut.activated.connect(self.quit_application)

    def quit_application(self):
        """Quit the application."""
        QCoreApplication.quit()

    def closeEvent(self, event):
        self.hide()
