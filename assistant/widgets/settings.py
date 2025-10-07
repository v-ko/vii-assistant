from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtGui import QGuiApplication, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from assistant.config import CLIENT_CONFIG
from assistant.util import encode_client_type
from assistant.view_states.terminal import TerminalViewState


class SettingsWidget(QWidget):
    """Widget for model settings with three columns."""

    single_step_clicked = Signal()  # Signal emitted when single step button is clicked
    # Signal emitted when clipboard step button is clicked
    clipboard_step_clicked = Signal()
    # Signal emitted when OCR clipboard button is clicked
    ocr_clipboard_clicked = Signal()
    # Session control signals
    start_session_clicked = Signal()
    stop_session_clicked = Signal()
    new_session_clicked = Signal()
    open_sessions_folder_clicked = Signal()

    def __init__(self, state: "TerminalViewState", parent=None):
        super().__init__(parent)
        self._state = state
        self._session_state = state.session_state
        self.setup_ui()
        self._bind_state()

    def setup_ui(self):
        """Set up the UI with three columns."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        # Slightly tighter spacing between the three columns
        main_layout.setSpacing(12)

        # First column: Session controls and capture actions
        first_column = QVBoxLayout()
        # Tighter vertical spacing for a denser layout
        first_column.setSpacing(8)

        # Session control buttons
        session_controls_layout = QHBoxLayout()
        session_controls_layout.setSpacing(6)
        # session_controls_layout.setContentsMargins(0, 0, 0, 0)

        self.start_session_button = QPushButton("▶")
        self.start_session_button.setToolTip("Begin recording a new session")
        self.start_session_button.clicked.connect(self.on_start_session_clicked)
        self.start_session_button.setAccessibleName("Start session")
        session_controls_layout.addWidget(self.start_session_button)

        self.stop_session_button = QPushButton("⏹")
        self.stop_session_button.setToolTip("Stop the current session")
        self.stop_session_button.clicked.connect(self.on_stop_session_clicked)
        self.stop_session_button.setAccessibleName("Stop session")
        session_controls_layout.addWidget(self.stop_session_button)

        self.new_session_button = QPushButton("New session")
        self.new_session_button.setToolTip("Create a new session workspace")
        self.new_session_button.clicked.connect(self.on_new_session_clicked)
        self.new_session_button.setAccessibleName("New session")
        session_controls_layout.addWidget(self.new_session_button)

        first_column.addLayout(session_controls_layout)

        # Normalize button heights
        uniform_button_height = 30
        for button in (
            self.start_session_button,
            self.stop_session_button,
            self.new_session_button,
        ):
            button.setMinimumWidth(48)
            button.setFixedHeight(uniform_button_height)
            button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        # Single step button with loading label
        step_layout = QHBoxLayout()
        self.single_step_button = QPushButton("Single step")
        self.single_step_button.setToolTip("Take a single screenshot and process it")
        self.single_step_button.clicked.connect(self.on_single_step_clicked)
        self.single_step_button.setFixedHeight(uniform_button_height)
        step_layout.addWidget(self.single_step_button)

        # Loading indicator icon (hidden by default)
        self.loading_icon = QLabel("⏳")  # Using hourglass emoji as static icon
        self.loading_icon.setVisible(False)
        self.loading_icon.setStyleSheet("font-size: 16px;")
        step_layout.addWidget(self.loading_icon)

        first_column.addLayout(step_layout)

        # Clipboard step button
        self.clipboard_step_button = QPushButton("Clipboard step")
        self.clipboard_step_button.setToolTip("Process image from clipboard")
        self.clipboard_step_button.clicked.connect(self.on_clipboard_step_clicked)
        self.clipboard_step_button.setFixedHeight(uniform_button_height)
        first_column.addWidget(self.clipboard_step_button)

        # OCR Clipboard button
        self.ocr_clipboard_button = QPushButton("OCR clipboard")
        self.ocr_clipboard_button.setToolTip(
            "Run OCR (Tesseract) on clipboard image and show text output"
        )
        self.ocr_clipboard_button.clicked.connect(self.on_ocr_clipboard_clicked)
        self.ocr_clipboard_button.setFixedHeight(uniform_button_height)
        first_column.addWidget(self.ocr_clipboard_button)
        first_column.addStretch()

        # Request in progress flag
        self._request_in_progress = False

        # Second column: Screen selector and Client selector
        second_column = QVBoxLayout()
        second_column.setSpacing(8)

        # Open sessions folder helper
        self.open_sessions_folder_button = QPushButton("Open sessions folder")
        self.open_sessions_folder_button.setToolTip(
            "Open the directory containing recorded sessions"
        )
        self.open_sessions_folder_button.clicked.connect(
            self.on_open_sessions_folder_clicked
        )
        self.open_sessions_folder_button.setFixedHeight(uniform_button_height)
        second_column.addWidget(self.open_sessions_folder_button)

        # Screen selector
        self.screen_combo = QComboBox()
        self.populate_screen_combo()
        self.screen_combo.currentTextChanged.connect(self.on_screen_changed)
        self.screen_combo.setFixedHeight(uniform_button_height)
        second_column.addWidget(self.screen_combo)

        # Client selector
        self.client_combo = QComboBox()
        self.populate_client_combo()
        self.client_combo.currentTextChanged.connect(self.on_client_changed)
        self.client_combo.setFixedHeight(uniform_button_height)
        second_column.addWidget(self.client_combo)

        second_column.addStretch()

        # Third column: Image placeholder
        third_column = QVBoxLayout()
        third_column.setSpacing(6)
        self.image_placeholder = QLabel("No image")
        self.image_placeholder.setFrameShape(QFrame.Shape.Box)
        self.image_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_placeholder.setMinimumSize(100, 100)
        third_column.addWidget(self.image_placeholder)
        third_column.addStretch()

        # Add columns to main layout
        main_layout.addLayout(first_column)
        main_layout.addLayout(second_column)
        main_layout.addLayout(third_column)

        # Apply styling
        self.setStyleSheet(
            """
            QWidget { background-color: #2d2d2d; color: #e0e0e0; }
            QComboBox {
                background-color: #3a3a3a;
                border: 1px solid #555;
                padding: 5px;
            }
        """
        )

        self._update_session_controls()

    def _bind_state(self):
        self._state.session_state_changed.connect(self._apply_session_state)
        self._state.client_type_changed.connect(self._apply_client_type)
        self._state.screen_changed.connect(self._apply_screen)
        self._state.request_in_progress_changed.connect(self._apply_request_in_progress)

        self._apply_session_state(self._state.session_state)
        self._apply_client_type(self._state.client_type)
        self._apply_screen(self._state.screen)
        self._apply_request_in_progress(self._state.request_in_progress)

    def populate_screen_combo(self):
        """Populate the screen combo box with available screens."""
        self.screen_combo.clear()
        screens = QGuiApplication.screens()
        for i, screen in enumerate(screens):
            geometry = screen.geometry()
            name = f"Screen {i+1}: {geometry.width()}x{geometry.height()}"
            self.screen_combo.addItem(name, screen.name())

    def populate_client_combo(self):
        """Populate the client combo with all available client:model combos."""
        self.client_combo.clear()
        for backend, models in CLIENT_CONFIG.items():
            for model in models:
                client_type = encode_client_type(backend, model)
                display_name = f"{backend.capitalize()} - {model}"
                self.client_combo.addItem(display_name, client_type)

    def on_start_session_clicked(self):
        """Handle start session button click."""
        if self._session_state == "started":
            return
        self.start_session_clicked.emit()

    def on_stop_session_clicked(self):
        """Handle stop session button click."""
        if self._session_state != "started":
            return
        self.stop_session_clicked.emit()

    def on_new_session_clicked(self):
        """Handle new session button click."""
        if self._session_state == "started":
            return
        self.new_session_clicked.emit()

    def on_open_sessions_folder_clicked(self):
        """Handle open sessions folder button click."""
        self.open_sessions_folder_clicked.emit()

    def on_screen_changed(self, screen_text):
        """Handle screen selection change."""
        index = self.screen_combo.currentIndex()
        if index >= 0:
            screen_name = self.screen_combo.itemData(index)
            if self._state.screen != screen_name:
                self._state.screen = screen_name

    def on_client_changed(self, client_text):
        """Handle client selection change."""
        index = self.client_combo.currentIndex()
        if index >= 0:
            client_type = self.client_combo.itemData(index)
            if self._state.client_type != client_type:
                self._state.client_type = client_type

    def _apply_session_state(self, value: str) -> None:
        if self._session_state == value:
            return
        self._session_state = value
        self._update_session_controls()

    def _update_session_controls(self) -> None:
        state = self._session_state
        self.start_session_button.setEnabled(state in {"new-session", "paused"})
        self.start_session_button.setToolTip(
            "Resume the current session"
            if state == "paused"
            else "Begin recording a new session"
        )
        self.stop_session_button.setEnabled(state == "started")
        self.new_session_button.setEnabled(state != "started")

        config_locked = state in {"started", "paused"}
        self.screen_combo.setEnabled(not config_locked)
        self.client_combo.setEnabled(not config_locked)

    def _apply_client_type(self, client_type: str) -> None:
        if not client_type:
            return
        for i in range(self.client_combo.count()):
            if self.client_combo.itemData(i) == client_type:
                blocker = QSignalBlocker(self.client_combo)
                self.client_combo.setCurrentIndex(i)
                break

    def _apply_screen(self, screen_name: str) -> None:
        if not screen_name:
            return
        for i in range(self.screen_combo.count()):
            if self.screen_combo.itemData(i) == screen_name:
                blocker = QSignalBlocker(self.screen_combo)
                self.screen_combo.setCurrentIndex(i)
                break

    def _apply_request_in_progress(self, value: bool) -> None:
        if self._request_in_progress == value:
            return
        self._request_in_progress = value
        self._set_request_in_progress_ui(value)

    def _set_request_in_progress_ui(self, value: bool) -> None:
        self.single_step_button.setEnabled(not value)
        self.clipboard_step_button.setEnabled(not value)
        self.ocr_clipboard_button.setEnabled(not value)
        self.loading_icon.setVisible(value)

    def on_single_step_clicked(self):
        """Handle single step button click."""
        if not self._request_in_progress:
            self.single_step_clicked.emit()

    def on_clipboard_step_clicked(self):
        """Handle clipboard step button click."""
        if not self._request_in_progress:
            self.clipboard_step_clicked.emit()

    def on_ocr_clipboard_clicked(self):
        """Handle OCR clipboard button click."""
        if not self._request_in_progress:
            self.ocr_clipboard_clicked.emit()

    @property
    def request_in_progress(self) -> bool:
        """Get the request in progress flag."""
        return self._request_in_progress

    @request_in_progress.setter
    def request_in_progress(self, value: bool):
        """Set the request in progress flag and update UI accordingly."""
        if self._request_in_progress == value:
            return
        self._request_in_progress = value
        self._set_request_in_progress_ui(value)
        if self._state.request_in_progress != value:
            self._state.request_in_progress = value

    def set_image(self, pixmap: QPixmap):
        """Set the image in the image placeholder."""
        if pixmap:
            # Scale the pixmap to fit the placeholder while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.image_placeholder.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.image_placeholder.setPixmap(scaled_pixmap)
        else:
            self.image_placeholder.setText("No image")
