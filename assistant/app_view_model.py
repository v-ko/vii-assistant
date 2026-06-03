"""App-level QML ViewModel.

Thin routing layer: translates QML @Slot calls into @action calls.
State lives in view state objects; mutation logic lives in
terminal_actions.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Property, QObject, Signal, Slot

from assistant.facade import vii
from assistant.model_configs import AVAILABLE_MODELS
from assistant.procedures import fetch_raw_context_and_present
from assistant.terminal_actions import (
    add_tool_prompt,
    apply_health_result,
    attach_clipboard,
    attach_screen,
    new_session,
    ocr_clipboard_action,
    open_app_config,
    open_experiment_config,
    open_focus_mode_prompt,
    open_sessions_folder,
    schedule_health_check,
    set_max_new_tokens,
    set_model,
    set_screen,
    step_experiment,
    stop_experiment,
    submit_message,
)

log = logging.getLogger(__name__)


class AppViewModel(QObject):
    """ViewModel that QML binds to. Routes slot calls to @action functions."""

    # Signal for async health/model check results
    health_check_done = Signal(bool, str, str)  # connected, model_state, model_key
    context_debug_ready = Signal(str, str)  # focus_mode, prompt_text
    screen_list_changed = Signal()
    primary_screen_changed = Signal()

    # Display-friendly server label (schema stripped)
    @Property(str, constant=True)
    def serverHost(self):
        return vii.inference_client.host_display

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.health_check_done.connect(self._apply_health_result)

    def bind_screens(self) -> None:
        """Connect to app_state.screens_changed to track primary screen."""
        vii.app.view_state.screens_changed.connect(self._on_screens_changed)

    def _on_screens_changed(self) -> None:
        self.primary_screen_changed.emit()
        self.screen_list_changed.emit()

    @Property(QObject, notify=primary_screen_changed)
    def primaryScreenInfo(self) -> QObject | None:
        return vii.app.view_state.primary_screen_info

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

    @Slot()
    def stopAssistant(self):
        from assistant.actions import stop_assistant

        stop_assistant()

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
    def setExecutionMode(self, mode: str):
        from assistant.facade import vii

        vii.app.view_state.settings_VS.execution_mode = mode

    @Slot(str)
    def setModel(self, model_key: str):
        set_model(model_key)

    @Slot(int)
    def setMaxNewTokens(self, value: int):
        set_max_new_tokens(value)

    @Slot()
    def addToolPrompt(self):
        add_tool_prompt()

    @Slot(str)
    def openFocusModePrompt(self, mode_name: str):
        open_focus_mode_prompt(mode_name)

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
        try:
            current = vii.experiments_manager.config_path
        except RuntimeError:
            current = None
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
        result = []
        for i, s in enumerate(vii.app.view_state.screens):
            display = f"Screen {i + 1}: {s.width}x{s.height}"
            result.append({"name": s.name, "displayName": display})
        return result

    # ── Debug ────────────────────────────────────────────────────

    @Slot()
    def showScreenDebug(self):
        from assistant.debug_actions import show_screen_debug

        show_screen_debug()

    @Slot(str)
    def fetchContextDebug(self, focus_mode: str):
        """Fetch the context prompt for a focus mode from the server."""
        fetch_raw_context_and_present(focus_mode).catch(
            lambda exc: log.error("fetchContextDebug failed: %s", exc)
        )

    # ── Health check ─────────────────────────────────────────────

    @Slot()
    def scheduleHealthCheck(self):
        schedule_health_check(self._on_health_result)

    def _on_health_result(self, connected: bool, model_state: str, model_key: str):
        self.health_check_done.emit(connected, model_state, model_key)

    def _apply_health_result(self, connected: bool, model_state: str, model_key: str):
        apply_health_result(connected, model_state, model_key)
