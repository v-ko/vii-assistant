"""QML ViewModel for the settings modal.

Exposes SettingsModalViewState properties and actions to QML.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fusion import get_logger
from PySide6.QtCore import Property, QObject, Signal, Slot

from assistant.facade import vii

if TYPE_CHECKING:
    from assistant.app_state import AppState

log = get_logger(__name__)


class SettingsModalViewModel(QObject):
    """ViewModel exposed to QML as 'settingsModalVM'."""

    visible_changed = Signal()
    sample_file_progress_changed = Signal()

    def __init__(self, app_state: AppState, parent: QObject | None = None) -> None:
        super().__init__(parent)
        vs = app_state.settings_modal_VS
        vs.visible_changed.connect(self.visible_changed)
        vs.transcription_section_vs.sample_file_progress_changed.connect(
            self.sample_file_progress_changed
        )

    # ── Properties ──

    @Property(bool, notify=visible_changed)
    def visible(self) -> bool:
        if vii._app_state is None:
            return False
        return vii.app_state.settings_modal_VS.visible

    @Property(float, notify=sample_file_progress_changed)
    def sampleFileProgress(self) -> float:
        if vii._app_state is None:
            return -1.0
        p = (
            vii.app_state.settings_modal_VS.transcription_section_vs.sample_file_progress
        )
        return p if p is not None else -1.0

    @Property(bool, notify=sample_file_progress_changed)
    def sampleFileInProgress(self) -> bool:
        if vii._app_state is None:
            return False
        return (
            vii.app_state.settings_modal_VS.transcription_section_vs.sample_file_progress
            is not None
        )

    # ── Slots ──

    @Slot()
    def show(self) -> None:
        vii.app_state.settings_modal_VS.visible = True

    @Slot()
    def hide(self) -> None:
        vii.app_state.settings_modal_VS.visible = False

    @Slot(str)
    def transcribeFile(self, file_path: str) -> None:
        """Start transcription of the given audio file."""
        from assistant.transcription_procedures import transcribe_file

        transcribe_file(file_path)
