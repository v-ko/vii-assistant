from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QCheckBox,
                               QComboBox, QLabel, QFrame, QPushButton)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication, QPixmap


class SettingsWidget(QWidget):
    """Widget for model settings with three columns."""

    config_changed = Signal(str, object)  # key, value
    single_step_clicked = Signal()  # Signal emitted when single step button is clicked

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Set up the UI with three columns."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(20)

        # First column: Auto-query checkbox and Single step button
        first_column = QVBoxLayout()

        # Auto-query checkbox
        self.auto_query_checkbox = QCheckBox("Auto-query")
        self.auto_query_checkbox.setToolTip("Enable automatic querying")
        self.auto_query_checkbox.stateChanged.connect(
            self.on_auto_query_changed)
        first_column.addWidget(self.auto_query_checkbox)

        # Single step button with loading label
        step_layout = QHBoxLayout()
        self.single_step_button = QPushButton("Single step")
        self.single_step_button.setToolTip("Take a single screenshot and process it")
        self.single_step_button.clicked.connect(self.on_single_step_clicked)
        step_layout.addWidget(self.single_step_button)

        # Loading label (hidden by default)
        self.loading_label = QLabel("Loading...")
        self.loading_label.setVisible(False)
        step_layout.addWidget(self.loading_label)

        first_column.addLayout(step_layout)
        first_column.addStretch()

        # Request in progress flag
        self._request_in_progress = False

        # Second column: Screen selector
        second_column = QVBoxLayout()
        self.screen_combo = QComboBox()
        self.populate_screen_combo()
        self.screen_combo.currentTextChanged.connect(self.on_screen_changed)
        second_column.addWidget(self.screen_combo)
        second_column.addStretch()

        # Third column: Image placeholder
        third_column = QVBoxLayout()
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
        self.setStyleSheet("""
            QWidget { background-color: #2d2d2d; color: #e0e0e0; }
            QComboBox {
                background-color: #3a3a3a;
                border: 1px solid #555;
                padding: 5px;
            }
        """)

    def populate_screen_combo(self):
        """Populate the screen combo box with available screens."""
        self.screen_combo.clear()
        screens = QGuiApplication.screens()
        for i, screen in enumerate(screens):
            geometry = screen.geometry()
            name = f"Screen {i+1}: {geometry.width()}x{geometry.height()}"
            self.screen_combo.addItem(name, screen.name())

    def on_auto_query_changed(self, state):
        """Handle auto-query checkbox state change."""
        is_checked = state == Qt.CheckState.Checked
        self.config_changed.emit("auto_query", is_checked)

    def on_screen_changed(self, screen_text):
        """Handle screen selection change."""
        index = self.screen_combo.currentIndex()
        if index >= 0:
            screen_name = self.screen_combo.itemData(index)
            self.config_changed.emit("screen", screen_name)

    def update_from_config(self, config):
        """Update widget state from config."""
        # Update auto-query checkbox
        auto_query = config.get("auto_query", False)
        self.auto_query_checkbox.setChecked(auto_query)

        # Update screen combo
        screen_name = config.get("screen", "")
        if screen_name:
            for i in range(self.screen_combo.count()):
                if self.screen_combo.itemData(i) == screen_name:
                    self.screen_combo.setCurrentIndex(i)
                    break

    def on_single_step_clicked(self):
        """Handle single step button click."""
        if not self._request_in_progress:
            self.single_step_clicked.emit()

    @property
    def request_in_progress(self) -> bool:
        """Get the request in progress flag."""
        return self._request_in_progress

    @request_in_progress.setter
    def request_in_progress(self, value: bool):
        """Set the request in progress flag and update UI accordingly."""
        self._request_in_progress = value
        self.single_step_button.setEnabled(not value)
        self.loading_label.setVisible(value)

    def set_image(self, pixmap: QPixmap):
        """Set the image in the image placeholder."""
        if pixmap:
            # Scale the pixmap to fit the placeholder while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.image_placeholder.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_placeholder.setPixmap(scaled_pixmap)
        else:
            self.image_placeholder.setText("No image")
