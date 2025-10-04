from PySide6.QtCore import (
    QCoreApplication,
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    Qt,
    Signal,
)
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from assistant.widgets.settings import SettingsWidget


class TerminalWindow(QWidget):
    # Signals
    system_prompt_changed = Signal(str)
    config_changed = Signal(str, object)  # key, value

    def __init__(self, parent=None):
        super().__init__(
            parent,
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint,
        )
        # Make the window background transparent
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setup_ui()
        self.position_window()
        self.setup_shortcuts()
        # self.setup_animations()

    def setup_ui(self):
        # Create a container widget to hold the UI; this widget will be
        # animated.
        self.container = QWidget(self)
        container_layout = QHBoxLayout(self.container)
        container_layout.setContentsMargins(10, 10, 10, 10)

        # Left side: system prompt text edit
        self.system_prompt_textedit = QTextEdit()
        self.system_prompt_textedit.setPlaceholderText("Enter system prompt here...")
        self.system_prompt_textedit.textChanged.connect(self._on_system_prompt_changed)
        # Set to accept only plain text (no formatting)
        self.system_prompt_textedit.setAcceptRichText(False)
        container_layout.addWidget(self.system_prompt_textedit, 1)

        # Right side: output label and model settings
        right_layout = QVBoxLayout()

        # Model settings widget
        self.model_settings = SettingsWidget()
        self.model_settings.config_changed.connect(self._on_config_changed)
        # single_step_clicked is connected in the app
        right_layout.addWidget(self.model_settings, 1)

        # Output text area (using QTextBrowser for selectable text)
        self.output_text = QTextBrowser()
        self.output_text.setText("Model output will appear here")
        self.output_text.setReadOnly(True)
        self.output_text.setFrameShape(QFrame.Shape.Box)
        self.output_text.setFrameShadow(QFrame.Shadow.Sunken)
        self.output_text.setOpenExternalLinks(False)
        right_layout.addWidget(self.output_text, 2)

        container_layout.addLayout(right_layout, 1)

        # Apply styling to the container only (the window is transparent)
        self.container.setStyleSheet(
            """
            QWidget {
                background-color: #2d2d2d;
                border: 1px solid #444;
                border-radius: 5px;
                color: #e0e0e0;
            }
            QTextEdit, QTextBrowser {
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px;
                color: #e0e0e0;
            }
        """
        )

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

    def set_output_text(self, text):
        self.output_text.setText(text)

    def get_system_prompt(self):
        return self.system_prompt_textedit.toPlainText()

    def _on_system_prompt_changed(self):
        """Handle system prompt text changes."""
        system_prompt = self.get_system_prompt()
        self.system_prompt_changed.emit(system_prompt)

    def _on_config_changed(self, key, value):
        """Handle config changes from the model settings widget."""
        if key == "system_prompt":
            # Update the system prompt text edit
            self.system_prompt_textedit.setPlainText(value)

        # Emit the config changed signal
        self.config_changed.emit(key, value)

    def update_from_config(self, config):
        """Update the UI from the current configuration."""
        # Update system prompt
        system_prompt = config.get("system_prompt", "")
        if system_prompt and system_prompt != self.get_system_prompt():
            self.system_prompt_textedit.setPlainText(system_prompt)

        # Update model settings widget
        self.model_settings.update_from_config(config)

    def setup_shortcuts(self):
        """Set up keyboard shortcuts."""
        self.alt_f4_shortcut = QShortcut(QKeySequence("Ctrl+w"), self)
        self.alt_f4_shortcut.activated.connect(self.quit_application)

    def quit_application(self):
        """Quit the application."""
        QCoreApplication.quit()

    def closeEvent(self, event):
        self.hide()
