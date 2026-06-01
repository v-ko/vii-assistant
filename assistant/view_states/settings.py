from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional

from fusion import get_logger
from fusion.platform.qt_widgets import Property
from PySide6.QtCore import QObject, Signal

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from assistant.services.project_manager import ViiProjectManager

from assistant.model_configs import DEFAULT_MODEL_KEY

log = get_logger(__name__)


class ExecutionMode(Enum):
    USER_APPROVE = "user-approve"
    AUTO = "auto"


_VALID_SESSION_STATES = {"new-session", "started", "paused", "disconnected", "error"}
_VALID_MODEL_STATES = {"unknown", "unloaded", "loading", "loaded"}


def _normalize_markdown(value: Optional[str]) -> str:
    if not value:
        return ""
    return value.rstrip("\n")


class AssistantSettingsViewState(QObject):
    session_state_changed = Signal(str)
    assistant_working_changed = Signal(bool)
    user_query_changed = Signal(str)
    system_prompt_changed = Signal(str)
    info_messages_changed = Signal(str)
    _info_message_enqueued = Signal(str)
    context_updates_allowed_changed = Signal(bool)
    selected_model_changed = Signal(str)
    server_model_state_changed = Signal(
        str
    )  # "unknown" | "unloaded" | "loading" | "loaded"
    server_model_key_changed = Signal(
        str
    )  # the model key actually loaded on the server
    execution_mode_changed = Signal(str)  # ExecutionMode.value
    max_new_tokens_changed = Signal(int)
    capture_screen_changed = Signal(str)
    experiment_configs_changed = Signal()
    selected_experiment_config_changed = Signal(str)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._project_manager: Optional["ViiProjectManager"] = None

        self._session_state = "new-session"
        self._assistant_working = False
        self._execution_mode = ExecutionMode.USER_APPROVE
        self._user_query = ""
        self._system_prompt = ""
        self._info_messages: list[str] = []
        self._context_updates_allowed = True
        self._selected_model = DEFAULT_MODEL_KEY
        self._server_model_state = "unknown"
        self._server_model_key = ""
        self._max_new_tokens = 256
        self._capture_screen = ""
        self._experiment_configs: list[dict] = []
        self._selected_experiment_config = ""
        self._info_message_enqueued.connect(self._append_info_message)

    # --- lifecycle -------------------------------------------------
    def reload_project_documents(self) -> None:
        if not self._project_manager:
            return
        task = _normalize_markdown(self._project_manager.get_task())
        prompt = _normalize_markdown(self._project_manager.get_system_prompt())
        self._set_user_query(task)
        self._set_system_prompt(prompt)

    # --- session_state ----------------------------------------------
    @Property(str, notify=session_state_changed)
    def session_state(self) -> str:
        return self._session_state

    @session_state.setter
    def session_state(self, value: str) -> None:
        if value not in _VALID_SESSION_STATES:
            raise ValueError(
                f"Invalid session state '{value}'. Expected one of"
                f" {_VALID_SESSION_STATES}."
            )
        if self._session_state == value:
            return
        self._session_state = value
        self.session_state_changed.emit(value)

    # --- assistant_working ------------------------------------------
    @Property(bool, notify=assistant_working_changed)
    def assistant_working(self) -> bool:
        return self._assistant_working

    @assistant_working.setter
    def assistant_working(self, value: bool) -> None:
        if self._assistant_working == value:
            return
        self._assistant_working = value
        log.info("assistant_working changed to %s", value)
        self.assistant_working_changed.emit(value)

    # --- execution_mode --------------------------------------------
    @Property(str, notify=execution_mode_changed)
    def execution_mode(self) -> str:
        return self._execution_mode.value

    @execution_mode.setter
    def execution_mode(self, value: str | ExecutionMode) -> None:
        if isinstance(value, str):
            value = ExecutionMode(value)
        if self._execution_mode == value:
            return
        self._execution_mode = value
        self.execution_mode_changed.emit(value.value)

    # --- user_query -------------------------------------------------
    @Property(str, notify=user_query_changed)
    def user_query_markdown(self) -> str:
        return self._user_query

    @user_query_markdown.setter
    def user_query_markdown(self, value: str) -> None:
        value = value or ""
        if self._user_query == value:
            return
        if not self._project_manager:
            raise RuntimeError(
                "SettingsViewState user_query_markdown set before project"
                " initialization"
            )
        self._project_manager.set_task(value)
        self._set_user_query(value)

    def _set_user_query(self, value: str) -> None:
        if self._user_query == value:
            return
        self._user_query = value
        self.user_query_changed.emit(value)

    # --- system_prompt ----------------------------------------------
    @Property(str, notify=system_prompt_changed)
    def system_prompt_markdown(self) -> str:
        return self._system_prompt

    @system_prompt_markdown.setter
    def system_prompt_markdown(self, value: str) -> None:
        value = value or ""
        if self._system_prompt == value:
            return
        if not self._project_manager:
            raise RuntimeError(
                "SettingsViewState system_prompt_markdown set before project"
                " initialization"
            )
        self._project_manager.set_system_prompt(value)
        self._set_system_prompt(value)

    def _set_system_prompt(self, value: str) -> None:
        if self._system_prompt == value:
            return
        self._system_prompt = value
        self.system_prompt_changed.emit(value)

    @Property(str, notify=info_messages_changed)
    def info_messages(self) -> str:
        return "\n".join(self._info_messages)

    def post_info_message(self, message: str) -> None:
        self._info_message_enqueued.emit(message)

    def clear_info_messages(self) -> None:
        if not self._info_messages:
            return
        self._info_messages.clear()
        self.info_messages_changed.emit("")

    def _append_info_message(self, message: str) -> None:
        msg = (message or "").strip()
        if not msg:
            return
        self._info_messages.append(msg)
        if len(self._info_messages) > 20:
            self._info_messages = self._info_messages[-20:]
        self.info_messages_changed.emit("\n".join(self._info_messages))

    @Property(bool, notify=context_updates_allowed_changed)
    def context_updates_allowed(self) -> bool:
        return self._context_updates_allowed

    @context_updates_allowed.setter
    def context_updates_allowed(self, value: bool) -> None:
        if self._context_updates_allowed == value:
            return
        self._context_updates_allowed = value
        self.context_updates_allowed_changed.emit(value)

    # --- selected_model -----------------------------------------------
    @Property(str, notify=selected_model_changed)
    def selected_model(self) -> str:
        return self._selected_model

    def _set_selected_model(self, value: str) -> None:
        if self._selected_model == value:
            return
        self._selected_model = value
        self.selected_model_changed.emit(value)

    # --- server_model_state (read from server health) ------------------
    @Property(str, notify=server_model_state_changed)
    def server_model_state(self) -> str:
        return self._server_model_state

    @server_model_state.setter
    def server_model_state(self, value: str) -> None:
        if value not in _VALID_MODEL_STATES:
            value = "unknown"
        if self._server_model_state == value:
            return
        self._server_model_state = value
        self.server_model_state_changed.emit(value)

    # --- server_model_key (which model is actually on the server) ------
    @Property(str, notify=server_model_key_changed)
    def server_model_key(self) -> str:
        return self._server_model_key

    @server_model_key.setter
    def server_model_key(self, value: str) -> None:
        if self._server_model_key == value:
            return
        self._server_model_key = value
        self.server_model_key_changed.emit(value)

    # --- max_new_tokens ------------------------------------------------
    @Property(int, notify=max_new_tokens_changed)
    def max_new_tokens(self) -> int:
        return self._max_new_tokens

    def _set_max_new_tokens(self, value: int) -> None:
        if self._max_new_tokens == value:
            return
        self._max_new_tokens = value
        self.max_new_tokens_changed.emit(value)

    # --- capture_screen ------------------------------------------------
    @Property(str, notify=capture_screen_changed)
    def capture_screen(self) -> str:
        return self._capture_screen

    def _set_capture_screen(self, value: str) -> None:
        if self._capture_screen == value:
            return
        self._capture_screen = value
        self.capture_screen_changed.emit(value)

    # --- experiment_configs -------------------------------------------
    @Property(list, notify=experiment_configs_changed)
    def experiment_configs(self) -> list[dict]:
        return self._experiment_configs

    def _set_experiment_configs(self, value: list[dict]) -> None:
        self._experiment_configs = value
        self.experiment_configs_changed.emit()

    # --- selected_experiment_config -----------------------------------
    @Property(str, notify=selected_experiment_config_changed)
    def selected_experiment_config(self) -> str:
        return self._selected_experiment_config

    @selected_experiment_config.setter
    def selected_experiment_config(self, value: str) -> None:
        if self._selected_experiment_config == value:
            return
        self._selected_experiment_config = value
        self.selected_experiment_config_changed.emit(value)
