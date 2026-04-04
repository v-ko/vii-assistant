import json
import threading
import urllib.error
import urllib.request

from PySide6.QtCore import QSignalBlocker, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QGuiApplication, QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from assistant.facade import facade
from assistant.model_configs import AVAILABLE_MODELS
from assistant.services.hybrid_segment_service import hfi
from assistant.services.project_manager import INFERENCE_HTTP_BASE
from assistant.view_states.settings import AssistantSettingsViewState


class SettingsWidget(QWidget):
    """Widget for model settings with three columns."""

    # Capture/action signals
    ocr_clipboard_clicked = Signal()
    attach_screen_clicked = Signal()
    attach_clipboard_clicked = Signal()
    # Session control signals
    new_session_clicked = Signal()
    open_sessions_folder_clicked = Signal()
    # Internal signal for cross-thread health check result
    # Carries (connected: bool, model_state: str, model_key: str)
    _health_check_result = Signal(bool, str, str)

    def __init__(self, state: AssistantSettingsViewState, parent=None):
        super().__init__(parent)
        self._state = state
        self._session_state = state.session_state
        self._context_updates_allowed = state.context_updates_allowed
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

        self.new_session_button = QPushButton("New session")
        self.new_session_button.setToolTip("Reset context and start a new session")
        self.new_session_button.clicked.connect(self.on_new_session_clicked)
        self.new_session_button.setAccessibleName("New session")
        session_controls_layout.addWidget(self.new_session_button)

        first_column.addLayout(session_controls_layout)

        # Normalize button heights
        uniform_button_height = 30
        for button in (self.new_session_button,):
            button.setMinimumWidth(48)
            button.setFixedHeight(uniform_button_height)
            button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        # OCR Clipboard button
        self.ocr_clipboard_button = QPushButton("OCR clipboard")
        self.ocr_clipboard_button.setToolTip(
            "Run OCR (Tesseract) on clipboard image and show text output"
        )
        self.ocr_clipboard_button.clicked.connect(self.on_ocr_clipboard_clicked)
        self.ocr_clipboard_button.setFixedHeight(uniform_button_height)
        first_column.addWidget(self.ocr_clipboard_button)

        self.info_messages_edit = QPlainTextEdit()
        self.info_messages_edit.setReadOnly(True)
        self.info_messages_edit.setPlaceholderText("Status messages")
        self.info_messages_edit.setMinimumHeight(80)
        self.info_messages_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        first_column.addWidget(self.info_messages_edit)

        self.attach_screen_button = QPushButton("Attach screen")
        self.attach_screen_button.setToolTip(
            "Capture the watched screen and add it to context"
        )
        self.attach_screen_button.clicked.connect(self.on_attach_screen_clicked)
        self.attach_screen_button.setFixedHeight(uniform_button_height)
        first_column.addWidget(self.attach_screen_button)

        self.attach_clipboard_button = QPushButton("Attach clipboard")
        self.attach_clipboard_button.setToolTip(
            "Add the current clipboard image to context"
        )
        self.attach_clipboard_button.clicked.connect(self.on_attach_clipboard_clicked)
        self.attach_clipboard_button.setFixedHeight(uniform_button_height)
        first_column.addWidget(self.attach_clipboard_button)

        # Server health status label
        self._health_label = QLabel("Server: checking...")
        self._health_label.setFixedHeight(uniform_button_height)
        self._health_label.setStyleSheet("QLabel { color: #888; }")
        first_column.addWidget(self._health_label)

        first_column.addStretch()

        # Request in progress flag
        self._request_in_progress = False

        # Health check timer (5 second interval, only pings when visible)
        self._health_check_result.connect(self._apply_health_result)
        self._health_timer = QTimer(self)
        self._health_timer.setInterval(5000)
        self._health_timer.timeout.connect(self._schedule_health_check)
        self._health_timer.start()
        self._schedule_health_check()

        # Second column: Model selector, screen selector
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

        self.open_app_config_button = QPushButton("Open app config")
        self.open_app_config_button.setToolTip(
            "Open the persisted app configuration file"
        )
        self.open_app_config_button.clicked.connect(self._on_open_app_config)
        self.open_app_config_button.setFixedHeight(uniform_button_height)
        second_column.addWidget(self.open_app_config_button)

        # Model load button + state label
        self.set_model_button = QPushButton("Set model")
        self.set_model_button.setFixedHeight(uniform_button_height)
        self.set_model_button.clicked.connect(self._on_set_model_clicked)
        second_column.addWidget(self.set_model_button)

        self._model_state_label = QLabel("Model: unknown")
        self._model_state_label.setFixedHeight(uniform_button_height)
        self._model_state_label.setStyleSheet("QLabel { color: #888; }")
        second_column.addWidget(self._model_state_label)

        # Screen selector
        self.screen_combo = QComboBox()
        self.populate_screen_combo()
        self.screen_combo.currentTextChanged.connect(self.on_screen_changed)
        self.screen_combo.setFixedHeight(uniform_button_height)
        second_column.addWidget(self.screen_combo)

        second_column.addStretch()

        # Third column: Project documents tabs
        third_column = QVBoxLayout()
        third_column.setSpacing(6)

        self.prompt_tabs = QTabWidget()
        self.prompt_tabs.setDocumentMode(True)

        self.user_query_edit = QPlainTextEdit()
        self.user_query_edit.setPlaceholderText("Describe the user request (markdown)")
        self.user_query_edit.textChanged.connect(self.on_user_query_changed)
        self.prompt_tabs.addTab(self.user_query_edit, "Notes")

        system_prompt_container = QWidget()
        sp_layout = QVBoxLayout(system_prompt_container)
        sp_layout.setContentsMargins(0, 0, 0, 0)
        sp_layout.setSpacing(4)
        self.system_prompt_edit = QPlainTextEdit()
        self.system_prompt_edit.setPlaceholderText(
            "Define the system prompt (markdown)"
        )
        self.system_prompt_edit.textChanged.connect(self.on_system_prompt_changed)
        sp_layout.addWidget(self.system_prompt_edit)
        self.add_tool_prompt_button = QPushButton("Add tool prompt")
        self.add_tool_prompt_button.clicked.connect(self._on_add_tool_prompt)
        sp_layout.addWidget(self.add_tool_prompt_button)
        self.prompt_tabs.addTab(system_prompt_container, "System Prompt")

        # Experiments tab
        experiments_container = QWidget()
        exp_layout = QVBoxLayout(experiments_container)
        exp_layout.setContentsMargins(0, 0, 0, 0)
        exp_layout.setSpacing(6)

        exp_buttons_layout = QHBoxLayout()
        self.step_experiment_button = QPushButton("Step")
        self.step_experiment_button.clicked.connect(self._on_step_experiment)
        exp_buttons_layout.addWidget(self.step_experiment_button)

        self.stop_experiment_button = QPushButton("Stop")
        self.stop_experiment_button.setEnabled(False)
        self.stop_experiment_button.clicked.connect(self._on_stop_experiment)
        exp_buttons_layout.addWidget(self.stop_experiment_button)

        self.open_config_button = QPushButton("Open Config")
        self.open_config_button.clicked.connect(self._on_open_experiment_config)
        exp_buttons_layout.addWidget(self.open_config_button)
        exp_layout.addLayout(exp_buttons_layout)

        self.experiment_status_label = QLabel("Status: idle")
        exp_layout.addWidget(self.experiment_status_label)
        exp_layout.addStretch()

        self.prompt_tabs.addTab(experiments_container, "Experiments")

        third_column.addWidget(self.prompt_tabs)

        # Add columns to main layout
        main_layout.addLayout(first_column)
        main_layout.addLayout(second_column)
        main_layout.addLayout(third_column, 2)

        self._update_session_controls()

    def _bind_state(self):
        self._state.session_state_changed.connect(self._apply_session_state)
        self._state.screen_changed.connect(self._apply_screen)
        self._state.request_in_progress_changed.connect(self._apply_request_in_progress)
        self._state.user_query_changed.connect(self._apply_user_query)
        self._state.system_prompt_changed.connect(self._apply_system_prompt)
        self._state.info_messages_changed.connect(self._apply_info_messages)
        self._state.context_updates_allowed_changed.connect(
            self._apply_context_updates_allowed
        )
        self._state.server_model_state_changed.connect(self._apply_server_model_state)
        self._state.server_model_key_changed.connect(self._apply_server_model_state)

        self._apply_session_state(self._state.session_state)
        self._apply_screen(self._state.screen)
        self._apply_request_in_progress(self._state.request_in_progress)
        self._apply_user_query(self._state.user_query_markdown)
        self._apply_system_prompt(self._state.system_prompt_markdown)
        self._apply_info_messages(self._state.info_messages)
        self._apply_context_updates_allowed(self._state.context_updates_allowed)

    def _apply_info_messages(self, text: str) -> None:
        blocker = QSignalBlocker(self.info_messages_edit)
        try:
            self.info_messages_edit.setPlainText(text)
        finally:
            del blocker
        if text:
            self.info_messages_edit.moveCursor(QTextCursor.MoveOperation.End)
            self.info_messages_edit.ensureCursorVisible()

    def populate_screen_combo(self):
        """Populate the screen combo box with available screens."""
        self.screen_combo.clear()
        screens = QGuiApplication.screens()
        for i, screen in enumerate(screens):
            geometry = screen.geometry()
            name = f"Screen {i+1}: {geometry.width()}x{geometry.height()}"
            self.screen_combo.addItem(name, screen.name())

    def populate_client_combo(self):  # pragma: no cover - placeholder
        """Client selection disabled; placeholder for future reintroduction."""
        return

    def on_new_session_clicked(self):
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

    def on_client_changed(self, client_text):  # pragma: no cover - placeholder
        return

    def _apply_session_state(self, value: str) -> None:
        if self._session_state == value:
            return
        self._session_state = value
        self._update_session_controls()

    def _update_session_controls(self) -> None:
        state = self._session_state
        self.new_session_button.setEnabled(state != "new-session")
        self.screen_combo.setEnabled(state not in {"started", "paused"})

    # client combo removed

    def _apply_client_type(
        self, client_type: str
    ) -> None:  # pragma: no cover - placeholder
        return

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
        self.ocr_clipboard_button.setEnabled(not value)
        self._refresh_attachment_buttons()

    def on_ocr_clipboard_clicked(self):
        """Handle OCR clipboard button click."""
        if not self._request_in_progress:
            self.ocr_clipboard_clicked.emit()

    def on_attach_screen_clicked(self):
        if not self._request_in_progress and self._context_updates_allowed:
            self.attach_screen_clicked.emit()

    def on_attach_clipboard_clicked(self):
        if not self._request_in_progress and self._context_updates_allowed:
            self.attach_clipboard_clicked.emit()

    def on_user_query_changed(self):
        text = self.user_query_edit.toPlainText()
        if text == self._state.user_query_markdown:
            return
        self._state.user_query_markdown = text

    def on_system_prompt_changed(self):
        text = self.system_prompt_edit.toPlainText()
        if text == self._state.system_prompt_markdown:
            return
        self._state.system_prompt_markdown = text

    def _on_add_tool_prompt(self) -> None:
        tool_prompt = hfi.generate_tool_prompt()
        if not tool_prompt:
            return
        current = self.system_prompt_edit.toPlainText()
        separator = "\n\n" if current.strip() else ""
        self.system_prompt_edit.setPlainText(current + separator + tool_prompt)

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

    def _apply_user_query(self, value: str) -> None:
        if value == self.user_query_edit.toPlainText():
            return
        blocker = QSignalBlocker(self.user_query_edit)
        self.user_query_edit.setPlainText(value)

    def _apply_system_prompt(self, value: str) -> None:
        if value == self.system_prompt_edit.toPlainText():
            return
        blocker = QSignalBlocker(self.system_prompt_edit)
        self.system_prompt_edit.setPlainText(value)

    def _apply_context_updates_allowed(self, allowed: bool) -> None:
        if self._context_updates_allowed != allowed:
            self._context_updates_allowed = allowed
        self._refresh_attachment_buttons()

    def _refresh_attachment_buttons(self) -> None:
        enabled = (not self._request_in_progress) and self._context_updates_allowed
        self.attach_screen_button.setEnabled(enabled)
        self.attach_clipboard_button.setEnabled(enabled)

    def _schedule_health_check(self) -> None:
        if not self.isVisible():
            return
        threading.Thread(target=self._do_health_check, daemon=True).start()

    def _do_health_check(self) -> None:
        url = f"{INFERENCE_HTTP_BASE}/status"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())
            model_info = data.get("model", {})
            model_state = model_info.get("state", "unknown")
            model_key = model_info.get("model_key") or ""
            self._health_check_result.emit(True, model_state, model_key)
        except Exception:
            self._health_check_result.emit(False, "unknown", "")

    def _apply_health_result(
        self, connected: bool, model_state: str, model_key: str
    ) -> None:
        # Update health label
        if connected:
            self._health_label.setText("Server: Connected")
            self._health_label.setStyleSheet("QLabel { color: #4CAF50; }")
        else:
            self._health_label.setText("Server: Disconnected")
            self._health_label.setStyleSheet("QLabel { color: #f44336; }")

        # Update model state in view state
        self._state.server_model_state = model_state
        self._state.server_model_key = model_key

    # --- Model dialog ---------------------------------------------------
    def _on_set_model_clicked(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Set model")
        layout = QVBoxLayout(dialog)

        combo = QComboBox()
        for key, display_name in AVAILABLE_MODELS.items():
            combo.addItem(display_name, key)
        # Pre-select the persisted selection
        current_key = self._state.selected_model
        for i in range(combo.count()):
            if combo.itemData(i) == current_key:
                combo.setCurrentIndex(i)
                break
        layout.addWidget(combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        model_key = combo.currentData()
        self._state.selected_model = model_key
        self._request_model_load(model_key)

    def _apply_server_model_state(self, *_args) -> None:
        state = self._state.server_model_state
        key = self._state.server_model_key
        if state == "loaded" and key:
            display = AVAILABLE_MODELS.get(key, key)
            self._model_state_label.setText(f"Model: {display}")
            self._model_state_label.setStyleSheet("QLabel { color: #4CAF50; }")
        elif state == "loading":
            self._model_state_label.setText("Model: loading...")
            self._model_state_label.setStyleSheet("QLabel { color: #FFA726; }")
        elif state == "unloaded":
            self._model_state_label.setText("Model: unloaded")
            self._model_state_label.setStyleSheet("QLabel { color: #888; }")
        else:
            self._model_state_label.setText("Model: unknown")
            self._model_state_label.setStyleSheet("QLabel { color: #888; }")

    def _request_model_load(self, model_key: str) -> None:
        threading.Thread(
            target=self._do_model_load, args=(model_key,), daemon=True
        ).start()

    def _do_model_load(self, model_key: str) -> None:
        url = f"{INFERENCE_HTTP_BASE}/model/load"
        payload = json.dumps({"model_key": model_key}).encode()
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            model_info = data.get("model", {})
            model_state = model_info.get("state", "unknown")
            mk = model_info.get("model_key") or ""
            self._health_check_result.emit(True, model_state, mk)
        except Exception:
            # Trigger a health check to get fresh state
            self._do_health_check()

    def _on_step_experiment(self):
        facade.experiments_manager.step()
        self._update_experiment_buttons()

    def _on_stop_experiment(self):
        facade.experiments_manager.stop()
        self._update_experiment_buttons()

    def _on_open_experiment_config(self):
        config_path = facade.experiments_manager.config_path
        if not config_path.exists():
            raise Exception(f"Experiment config not found: {config_path}")
        print(f"Opening experiment config: {config_path}")
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(config_path)))

    def _on_open_app_config(self):
        config_path = facade.config.config_file
        if not config_path.exists():
            raise Exception(f"App config not found: {config_path}")
        print(f"Opening app config: {config_path}")
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(config_path)))

    def _update_experiment_buttons(self):
        running = facade.experiments_manager.running
        self.stop_experiment_button.setEnabled(running)
        self.experiment_status_label.setText(
            f"Status: {facade.experiments_manager.state.value}"
            f" (step {facade.experiments_manager._current_step})"
        )
