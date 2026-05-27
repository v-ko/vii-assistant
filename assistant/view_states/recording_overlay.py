from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class RecordingOverlayViewState(QObject):
    """State for the recording overlay UI.

    Contains only mode flags — no spectrum data.
    Spectrum goes directly to the view model (transient visualization).
    """

    recording_active_changed = Signal(bool)
    transcribing_active_changed = Signal(bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._recording_active: bool = False
        self._transcribing_active: bool = False

    @property
    def recording_active(self) -> bool:
        return self._recording_active

    @recording_active.setter
    def recording_active(self, value: bool) -> None:
        if self._recording_active == value:
            return
        self._recording_active = value
        self.recording_active_changed.emit(value)

    @property
    def transcribing_active(self) -> bool:
        return self._transcribing_active

    @transcribing_active.setter
    def transcribing_active(self, value: bool) -> None:
        if self._transcribing_active == value:
            return
        self._transcribing_active = value
        self.transcribing_active_changed.emit(value)

    @property
    def ui_mode(self) -> str:
        """Derived display mode: 'recording' | 'transcribing' | 'idle'."""
        if self._recording_active:
            return "recording"
        if self._transcribing_active:
            return "transcribing"
        return "idle"
