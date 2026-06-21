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
from assistant.procedures import fetch_raw_context_and_present, run_all_experiment
from assistant.terminal_actions import (
    add_tool_prompt,
    attach_clipboard,
    attach_screen,
    cancel_experiment,
    generate_experiment_stats,
    new_session,
    ocr_clipboard_action,
    open_app_config,
    open_experiment_config,
    open_focus_mode_prompt,
    open_sessions_folder,
    random_experiment_step,
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

    context_debug_ready = Signal(str, str)  # focus_mode, prompt_text
    screen_list_changed = Signal()
    primary_screen_changed = Signal()
    experiment_progress = Signal(int, int)  # current_step, total
    active_agent_changed = Signal(str)

    # Display-friendly server label (schema stripped)
    @Property(str, constant=True)
    def serverHost(self):
        return vii.inference_client.host_display

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)

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
        set_model(model_key).catch(lambda exc: log.error("setModel failed: %s", exc))

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
    def randomExperimentStep(self):
        random_experiment_step()

    @Slot()
    def stopExperiment(self):
        stop_experiment()

    @Slot()
    def runAllExperiment(self):
        task = run_all_experiment()
        vii.experiments_manager._run_all_task = task

    @Slot()
    def cancelExperiment(self):
        cancel_experiment()

    @Slot()
    def openExperimentConfig(self):
        open_experiment_config()

    @Slot()
    def generateExperimentStats(self):
        generate_experiment_stats()

    @Slot(result=list)
    def getExperimentConfigs(self) -> list:
        from assistant.constants import EXPERIMENTS_DIR

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
    def getAgents(self) -> list:
        active = vii.active_agent
        return [
            {"name": name, "selected": name == active}
            for name in vii.get_available_agents()
        ]

    @Property(str, notify=active_agent_changed)
    def activeAgent(self) -> str:
        return vii.active_agent or ""

    @Slot(str)
    def setActiveAgent(self, name: str):
        vii.set_active_agent(name)
        self.active_agent_changed.emit(name)

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
