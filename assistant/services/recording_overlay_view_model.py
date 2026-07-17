"""
My greatest thanks to the developers of Handy (https://github.com/cjpais/Handy) for
inspiring this UI design for recording feedback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Property, QObject, Signal

if TYPE_CHECKING:
    from assistant.app_state import AppViewState
    from assistant.view_states.recording_overlay import RecordingOverlayViewState


class RecordingOverlayViewModel(QObject):
    """ViewModel exposed to QML as 'recordingOverlayVM'."""

    bars_changed = Signal()
    has_view_state_changed = Signal()
    ui_mode_changed = Signal()
    transcribing_active_changed = Signal()

    def __init__(self, app_state: AppViewState, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._bar_levels: list[float] = [0.0] * 10
        self._has_view_state: bool = False
        self._ui_mode: str = "idle"
        self._transcribing_active: bool = False
        self._view_state: RecordingOverlayViewState | None = None
        self._app_state = app_state
        app_state.recording_overlay_VS_changed.connect(self._on_vs_lifecycle)

    # ── Bar levels (transient spectrum data, not in ViewState) ──

    def set_bar_levels(self, levels: list[float]) -> None:
        """Called by AudioRecordingService at 10 Hz."""
        self._bar_levels = levels
        self.bars_changed.emit()

    @Property("QVariantList", notify=bars_changed)
    def bar_levels(self) -> list[float]:
        return self._bar_levels

    # ── ViewState proxy properties ──

    @Property(bool, notify=has_view_state_changed)
    def has_view_state(self) -> bool:
        return self._has_view_state

    @Property(str, notify=ui_mode_changed)
    def ui_mode(self) -> str:
        return self._ui_mode

    @Property(bool, notify=transcribing_active_changed)
    def transcribing_active(self) -> bool:
        return self._transcribing_active

    # ── Reactive binding to ViewState lifecycle ──

    def _on_vs_lifecycle(self) -> None:
        """Called when recording_overlay_VS is created or destroyed."""
        new_vs = self._app_state.recording_overlay_VS
        old_vs = self._view_state

        if old_vs is not None:
            old_vs.recording_active_changed.disconnect(self._sync)
            old_vs.transcribing_active_changed.disconnect(self._sync)

        self._view_state = new_vs

        if new_vs is not None:
            new_vs.recording_active_changed.connect(self._sync)
            new_vs.transcribing_active_changed.connect(self._sync)

        self._sync()

    def _sync(self) -> None:
        """Derive QML-facing properties from current view state."""
        vs = self._view_state
        new_has_vs = vs is not None
        new_mode = vs.ui_mode if vs else "idle"
        new_transcribing = vs.transcribing_active if vs else False

        if new_has_vs != self._has_view_state:
            self._has_view_state = new_has_vs
            self.has_view_state_changed.emit()

        if new_mode != self._ui_mode:
            self._ui_mode = new_mode
            self.ui_mode_changed.emit()

        if new_transcribing != self._transcribing_active:
            self._transcribing_active = new_transcribing
            self.transcribing_active_changed.emit()

        # Clear bars when overlay hidden
        if not new_has_vs and any(b != 0.0 for b in self._bar_levels):
            self._bar_levels = [0.0] * 10
            self.bars_changed.emit()
