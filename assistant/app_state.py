from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from assistant.view_states.context_view import ContextViewerState
from assistant.view_states.overlay import OverlayViewState
from assistant.view_states.recording_overlay import RecordingOverlayViewState
from assistant.view_states.settings import AssistantSettingsViewState
from assistant.view_states.settings_modal import SettingsModalViewState
from assistant.view_states.snippet import SnippetOverlayViewState


class AppState(QObject):
    """Container for top-level UI/application view states."""

    recording_overlay_VS_changed = Signal()
    snippet_overlays_changed = Signal()

    settings_VS: AssistantSettingsViewState
    context_VS: ContextViewerState
    overlay_VS: OverlayViewState
    settings_modal_VS: SettingsModalViewState
    snippet_overlays: list[SnippetOverlayViewState]

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.settings_VS = AssistantSettingsViewState(parent=self)
        self.context_VS = ContextViewerState(parent=self)
        self.overlay_VS = OverlayViewState(parent=self)
        self._recording_overlay_VS: RecordingOverlayViewState | None = None
        self.settings_modal_VS = SettingsModalViewState(parent=self)
        self.snippet_overlays = []
        self.settings_VS.context_updates_allowed_changed.connect(
            self._apply_context_controls
        )
        self._apply_context_controls(self.settings_VS.context_updates_allowed)

    @property
    def recording_overlay_VS(self) -> RecordingOverlayViewState | None:
        return self._recording_overlay_VS

    @recording_overlay_VS.setter
    def recording_overlay_VS(self, value: RecordingOverlayViewState | None) -> None:
        if self._recording_overlay_VS is value:
            return
        self._recording_overlay_VS = value
        self.recording_overlay_VS_changed.emit()

    def _apply_context_controls(self, allowed: bool) -> None:
        self.context_VS.interactions_enabled = allowed
