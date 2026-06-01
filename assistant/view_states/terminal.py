from __future__ import annotations

from fusion.platform.qt_widgets import Property
from PySide6.QtCore import QObject, Signal

from assistant.app_state import AppViewState


class TerminalViewState(QObject):
    """Qt-backed state for the assistant terminal window."""

    visible_changed = Signal(bool)
    output_text_changed = Signal(str)

    def __init__(self, app_state: AppViewState, parent: QObject | None = None):
        super().__init__(parent)
        self._visible = False
        self._output_text = ""
        self.app_state = app_state
        self.settings = app_state.settings_VS
        self.context_view = app_state.context_VS

    # --- visible ---
    @Property(bool, notify=visible_changed)
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        if self._visible == value:
            return
        self._visible = value
        self.visible_changed.emit(value)

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
