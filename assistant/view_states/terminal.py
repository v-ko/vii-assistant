from __future__ import annotations

from fusion.platform.qt_widgets import Property
from PySide6.QtCore import QObject, Signal

from assistant.app_state import AppState


class TerminalViewState(QObject):
    """Qt-backed state for the assistant terminal window."""

    output_text_changed = Signal(str)

    def __init__(self, app_state: AppState, parent: QObject | None = None):
        super().__init__(parent)
        self._output_text = ""
        self.app_state = app_state
        self.settings = app_state.settings
        self.context_view = app_state.context

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
