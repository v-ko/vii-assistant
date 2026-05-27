from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class TranscriptionSectionViewState(QObject):
    """State for the transcription section in the settings modal."""

    sample_file_progress_changed = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._sample_file_progress: float | None = None

    @property
    def sample_file_progress(self) -> float | None:
        return self._sample_file_progress

    @sample_file_progress.setter
    def sample_file_progress(self, value: float | None) -> None:
        if self._sample_file_progress == value:
            return
        self._sample_file_progress = value
        self.sample_file_progress_changed.emit()


class SettingsModalViewState(QObject):
    """Top-level view state for the settings modal."""

    visible_changed = Signal(bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.transcription_section_vs = TranscriptionSectionViewState(parent=self)
        self._visible: bool = False

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        if self._visible == value:
            return
        self._visible = value
        self.visible_changed.emit(value)
