from __future__ import annotations

from PySide6.QtCore import QObject

from assistant.view_states.context_view import ContextViewerState
from assistant.view_states.overlay import OverlayViewState
from assistant.view_states.settings import AssistantSettingsViewState


class AppState(QObject):
    """Container for top-level UI/application view states."""

    settings_VS: AssistantSettingsViewState
    context_VS: ContextViewerState
    overlay_VS: OverlayViewState

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.settings_VS = AssistantSettingsViewState(parent=self)
        self.context_VS = ContextViewerState(parent=self)
        self.overlay_VS = OverlayViewState(parent=self)
        self.settings_VS.context_updates_allowed_changed.connect(
            self._apply_context_controls
        )
        self._apply_context_controls(self.settings_VS.context_updates_allowed)

    def _apply_context_controls(self, allowed: bool) -> None:
        self.context_VS.interactions_enabled = allowed
