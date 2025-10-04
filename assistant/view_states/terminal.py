from __future__ import annotations

from typing import Any, Dict

from fusion.platform.qt_widgets import Property
from fusion.platform.qt_widgets.view_state import QtViewState
from PySide6.QtCore import Signal


class TerminalViewState(QtViewState):
    """Qt-backed state for the assistant terminal window."""

    system_prompt_changed = Signal(str)
    output_text_changed = Signal(str)
    auto_query_changed = Signal(bool)
    client_type_changed = Signal(str)
    screen_changed = Signal(str)
    request_in_progress_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._system_prompt = ""
        self._output_text = "Model output will appear here"
        self._auto_query = False
        self._client_type = "ollama:moondream"
        self._screen = ""
        self._request_in_progress = False

    # --- system_prompt ---
    @Property(str, notify=system_prompt_changed)
    def system_prompt(self) -> str:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        if self._system_prompt == value:
            return
        self._system_prompt = value
        self.system_prompt_changed.emit(value)

    # --- output_text ---
    @Property(str, notify=output_text_changed)
    def output_text(self) -> str:
        return self._output_text

    @output_text.setter
    def output_text(self, value: str) -> None:
        if self._output_text == value:
            return
        self._output_text = value
        self.output_text_changed.emit(value)

    # --- auto_query ---
    @Property(bool, notify=auto_query_changed)
    def auto_query(self) -> bool:
        return self._auto_query

    @auto_query.setter
    def auto_query(self, value: bool) -> None:
        if self._auto_query == value:
            return
        self._auto_query = value
        self.auto_query_changed.emit(value)

    # --- client_type ---
    @Property(str, notify=client_type_changed)
    def client_type(self) -> str:
        return self._client_type

    @client_type.setter
    def client_type(self, value: str) -> None:
        if self._client_type == value:
            return
        self._client_type = value
        self.client_type_changed.emit(value)

    # --- screen ---
    @Property(str, notify=screen_changed)
    def screen(self) -> str:
        return self._screen

    @screen.setter
    def screen(self, value: str) -> None:
        if self._screen == value:
            return
        self._screen = value
        self.screen_changed.emit(value)

    # --- request_in_progress ---
    @Property(bool, notify=request_in_progress_changed)
    def request_in_progress(self) -> bool:
        return self._request_in_progress

    @request_in_progress.setter
    def request_in_progress(self, value: bool) -> None:
        if self._request_in_progress == value:
            return
        self._request_in_progress = value
        self.request_in_progress_changed.emit(value)

    # --- Helpers ---
    def update_from_config(self, config: Dict[str, Any]) -> None:
        self.system_prompt = config.get("system_prompt", self._system_prompt)
        self.auto_query = config.get("auto_query", self._auto_query)
        self.client_type = config.get("client_type", self._client_type)
        self.screen = config.get("screen", self._screen)
