from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

from fusion.platform.qt_widgets import Property
from PySide6.QtCore import QObject, Signal

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from assistant.config import Config
    from assistant.services.project_manager import ViiProjectManager

_VALID_SESSION_STATES = {"new-session", "started", "paused"}


def _normalize_markdown(value: Optional[str]) -> str:
    if not value:
        return ""
    return value.rstrip("\n")


class SettingsViewState(QObject):
    session_state_changed = Signal(str)
    screen_changed = Signal(str)
    request_in_progress_changed = Signal(bool)
    user_query_changed = Signal(str)
    system_prompt_changed = Signal(str)
    info_messages_changed = Signal(str)
    _info_message_enqueued = Signal(str)
    context_updates_allowed_changed = Signal(bool)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        # External services (not used for direct persistence anymore; config persistence
        # service listens to our signals and performs debounced writes)
        self._config: Optional["Config"] = None
        self._project_manager: Optional["ViiProjectManager"] = None

        self._session_state = "new-session"
        # client_type removed (single hardcoded backend)
        self._screen = ""
        self._request_in_progress = False
        self._user_query = ""
        self._system_prompt = ""
        self._info_messages: list[str] = []
        self._context_updates_allowed = True
        self._info_message_enqueued.connect(self._append_info_message)

    # --- lifecycle -------------------------------------------------
    def initialize(
        self,
        config: "Config",
        project_manager: "ViiProjectManager",
    ) -> None:
        self._config = config
        self._project_manager = project_manager
        self.apply_config(config.data())
        self.reload_project_documents()

    def apply_config(self, config: Dict[str, object]) -> None:
        screen = str(config.get("screen", self._screen) or "")
        self._set_screen(screen)

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

    # client_type removed

    # --- screen -----------------------------------------------------
    @Property(str, notify=screen_changed)
    def screen(self) -> str:
        return self._screen

    @screen.setter
    def screen(self, value: str) -> None:
        if self._screen == value:
            return
        self._set_screen(value)

    def _set_screen(self, value: str) -> None:
        if self._screen == value:
            return
        self._screen = value
        self.screen_changed.emit(value)

    # --- request_in_progress ---------------------------------------
    @Property(bool, notify=request_in_progress_changed)
    def request_in_progress(self) -> bool:
        return self._request_in_progress

    @request_in_progress.setter
    def request_in_progress(self, value: bool) -> None:
        if self._request_in_progress == value:
            return
        self._request_in_progress = value
        self.request_in_progress_changed.emit(value)

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
