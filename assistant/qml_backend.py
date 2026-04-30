"""Python backend (controller) for the QML-based terminal window.

Thin routing layer: translates QML @Slot calls into @action calls.
State lives in view state objects; mutation logic lives in
terminal_actions.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QGuiApplication

from assistant.facade import vii
from assistant.model_configs import AVAILABLE_MODELS
from assistant.terminal_actions import (
    add_tool_prompt,
    attach_clipboard,
    attach_screen,
    new_session,
    ocr_clipboard_action,
    open_app_config,
    open_experiment_config,
    open_sessions_folder,
    schedule_health_check,
    set_model,
    set_screen,
    step_experiment,
    stop_experiment,
    submit_message,
)


class QmlBackend(QObject):
    """Controller that QML binds to. Routes slot calls to @action functions."""

    # Signal for async health/model check results
    health_check_done = Signal(bool, str, str)  # connected, model_state, model_key
    screen_list_changed = Signal()

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.health_check_done.connect(self._apply_health_result)

    # ── Session / message slots ──────────────────────────────────

    @Slot()
    def newSession(self):
        new_session()

    @Slot()
    def openSessionsFolder(self):
        open_sessions_folder()

    @Slot(str)
    def submitMessage(self, text: str):
        submit_message(text)

    # ── Capture slots ────────────────────────────────────────────

    @Slot()
    def ocrClipboard(self):
        ocr_clipboard_action()

    @Slot()
    def attachScreen(self):
        attach_screen()

    @Slot()
    def attachClipboard(self):
        attach_clipboard()

    # ── Settings slots ───────────────────────────────────────────

    @Slot(str)
    def setScreen(self, screen_name: str):
        set_screen(screen_name)

    @Slot(str)
    def setModel(self, model_key: str):
        set_model(model_key)

    @Slot()
    def addToolPrompt(self):
        add_tool_prompt()

    @Slot()
    def openAppConfig(self):
        open_app_config()

    # ── Experiment slots ─────────────────────────────────────────

    @Slot()
    def stepExperiment(self):
        step_experiment()

    @Slot()
    def stopExperiment(self):
        stop_experiment()

    @Slot()
    def openExperimentConfig(self):
        open_experiment_config()

    @Slot(result=list)
    def getExperimentConfigs(self) -> list:
        from assistant.experiments_manager import EXPERIMENTS_DIR

        configs = sorted(EXPERIMENTS_DIR.glob("*.json"))
        current = vii.experiments_manager.config_path
        return [
            {"name": p.stem, "path": str(p), "selected": p == current} for p in configs
        ]

    @Slot(str)
    def setExperimentConfig(self, path: str):
        from pathlib import Path

        vii.experiments_manager.config_path = Path(path)

    # ── Data queries (non-mutating, no @action needed) ───────────

    @Slot(result=list)
    def getAvailableModels(self) -> list:
        return [{"key": k, "displayName": v} for k, v in AVAILABLE_MODELS.items()]

    @Slot(result=list)
    def getScreenList(self) -> list:
        screens = QGuiApplication.screens()
        result = []
        for i, screen in enumerate(screens):
            geo = screen.geometry()
            display = f"Screen {i + 1}: {geo.width()}x{geo.height()}"
            result.append({"name": screen.name(), "displayName": display})
        return result

    # ── Health check ─────────────────────────────────────────────

    @Slot()
    def scheduleHealthCheck(self):
        schedule_health_check(self._on_health_result)

    def _on_health_result(self, connected: bool, model_state: str, model_key: str):
        self.health_check_done.emit(connected, model_state, model_key)

    def _apply_health_result(self, connected: bool, model_state: str, model_key: str):
        settings = vii.app_state.settings_VS
        settings.server_model_state = model_state
        settings.server_model_key = model_key
