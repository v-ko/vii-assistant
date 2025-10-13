from __future__ import annotations

from PySide6.QtCore import QObject

from assistant.view_states.context_view import ContextViewerState
from assistant.view_states.settings import SettingsViewState


class AppState(QObject):
    """Container for top-level UI/application view states."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.settings = SettingsViewState(parent=self)
        self.context = ContextViewerState(parent=self)
