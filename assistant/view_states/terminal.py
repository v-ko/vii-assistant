from __future__ import annotations

from fusion.platform.qt_widgets import Property
from PySide6.QtCore import QObject, Signal

from assistant.app_state import AppState


class TerminalViewState(QObject):
    """Qt-backed state for the assistant terminal window."""

    output_text_changed = Signal(str)
    geometry_changed = Signal()

    def __init__(self, app_state: AppState, parent: QObject | None = None):
        super().__init__(parent)
        self._output_text = ""
        self._win_x = 0
        self._win_y = 0
        self._win_width = 800
        self._win_height = 400
        self.app_state = app_state
        self.settings = app_state.settings_VS
        self.context_view = app_state.context_VS

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

    # --- Window geometry (set by screen config action, bound by QML) ---
    @Property(int, notify=geometry_changed)
    def win_x(self) -> int:
        return self._win_x

    @Property(int, notify=geometry_changed)
    def win_y(self) -> int:
        return self._win_y

    @Property(int, notify=geometry_changed)
    def win_width(self) -> int:
        return self._win_width

    @Property(int, notify=geometry_changed)
    def win_height(self) -> int:
        return self._win_height

    def set_geometry(self, x: int, y: int, w: int, h: int) -> None:
        """Update window geometry. Emits geometry_changed if anything changed."""
        changed = (
            x != self._win_x
            or y != self._win_y
            or w != self._win_width
            or h != self._win_height
        )
        if not changed:
            return
        self._win_x = x
        self._win_y = y
        self._win_width = w
        self._win_height = h
        self.geometry_changed.emit()
