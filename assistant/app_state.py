from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from assistant.view_states.context_view import ContextViewerState
from assistant.view_states.overlay import OverlayViewState
from assistant.view_states.recording_overlay import RecordingOverlayViewState
from assistant.view_states.screen_info import ScreenInfoVS
from assistant.view_states.settings import AssistantSettingsViewState
from assistant.view_states.settings_modal import SettingsModalViewState
from assistant.view_states.snippet import SnippetOverlayViewState


class AppViewState(QObject):
    """Container for top-level UI/application view states."""

    recording_overlay_VS_changed = Signal()
    snippet_overlays_changed = Signal()
    screen_debug_visible_changed = Signal(bool)
    screens_changed = Signal()

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
        self._screens: list[ScreenInfoVS] = []
        self._screen_debug_visible: bool = False
        self.settings_VS.context_updates_allowed_changed.connect(
            self._apply_context_controls
        )
        self._apply_context_controls(self.settings_VS.context_updates_allowed)

    # --- Screens ---

    @property
    def screens(self) -> list[ScreenInfoVS]:
        return self._screens

    @screens.setter
    def screens(self, value: list[ScreenInfoVS]) -> None:
        self._screens = value
        self.screens_changed.emit()

    @property
    def primary_screen_info(self) -> ScreenInfoVS | None:
        for s in self._screens:
            if s.is_primary:
                return s
        return None

    @property
    def capture_screen_info(self) -> ScreenInfoVS | None:
        for s in self._screens:
            if s.is_capture:
                return s
        return None

    @property
    def screen_debug_visible(self) -> bool:
        return self._screen_debug_visible

    @screen_debug_visible.setter
    def screen_debug_visible(self, value: bool) -> None:
        if self._screen_debug_visible == value:
            return
        self._screen_debug_visible = value
        self.screen_debug_visible_changed.emit(value)

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
