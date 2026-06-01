"""QML ViewModel for the settings modal.

Exposes SettingsModalViewState properties and actions to QML.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fusion import get_logger
from PySide6.QtCore import Property, QObject, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices

from assistant.constants import RECORDINGS_DIR
from assistant.facade import vii
from assistant.services.audio_recording import AudioRecordingService
from assistant.terminal_actions import set_input_device, set_settings_modal_visible
from assistant.transcription_procedures import transcribe_file

if TYPE_CHECKING:
    from assistant.app_state import AppViewState

log = get_logger(__name__)


class SettingsModalViewModel(QObject):
    """ViewModel exposed to QML as 'settingsModalVM'."""

    visible_changed = Signal()
    sample_file_progress_changed = Signal()
    available_input_devices_changed = Signal()
    selected_input_device_index_changed = Signal()

    def __init__(self, app_state: AppViewState, parent: QObject | None = None) -> None:
        super().__init__(parent)
        vs = app_state.settings_modal_VS
        vs.visible_changed.connect(self.visible_changed)
        vs.transcription_section_vs.sample_file_progress_changed.connect(
            self.sample_file_progress_changed
        )
        vs.transcription_section_vs.available_input_devices_changed.connect(
            self.available_input_devices_changed
        )
        vs.transcription_section_vs.selected_input_device_changed.connect(
            self._on_selected_device_changed
        )
        vs.visible_changed.connect(self._on_visible_changed)

    def _on_visible_changed(self, visible: bool) -> None:
        if visible:
            self._refresh_input_devices()

    def _on_selected_device_changed(self) -> None:
        self.selected_input_device_index_changed.emit()

    def _refresh_input_devices(self) -> None:
        """Refresh the list of available input devices."""
        devices = AudioRecordingService.available_input_devices()
        ts_vs = vii.app.view_state.settings_modal_VS.transcription_section_vs
        ts_vs.available_input_devices = devices
        self.available_input_devices_changed.emit()
        self.selected_input_device_index_changed.emit()

    # ── Properties ──

    @Property(bool, notify=visible_changed)
    def visible(self) -> bool:
        return vii.app.view_state.settings_modal_VS.visible

    @Property(float, notify=sample_file_progress_changed)
    def sampleFileProgress(self) -> float:
        p = (
            vii.app.view_state.settings_modal_VS.transcription_section_vs.sample_file_progress
        )
        return p if p is not None else -1.0

    @Property(bool, notify=sample_file_progress_changed)
    def sampleFileInProgress(self) -> bool:
        return (
            vii.app.view_state.settings_modal_VS.transcription_section_vs.sample_file_progress
            is not None
        )

    @Property(list, notify=available_input_devices_changed)
    def inputDeviceNames(self) -> list[str]:
        """Return device descriptions for the ComboBox model (with 'System Default' first)."""
        devices = (
            vii.app.view_state.settings_modal_VS.transcription_section_vs.available_input_devices
        )
        return ["System Default"] + [d["description"] for d in devices]

    @Property(int, notify=selected_input_device_index_changed)
    def selectedInputDeviceIndex(self) -> int:
        """Index into inputDeviceNames (0 = System Default)."""
        ts_vs = vii.app.view_state.settings_modal_VS.transcription_section_vs
        selected_id = ts_vs.selected_input_device
        if not selected_id:
            return 0
        for i, dev in enumerate(ts_vs.available_input_devices):
            if dev["id"] == selected_id:
                return i + 1  # +1 because index 0 is "System Default"
        return 0

    # ── Slots ──

    @Slot()
    def show(self) -> None:
        set_settings_modal_visible(True)

    @Slot()
    def hide(self) -> None:
        set_settings_modal_visible(False)

    @Slot(int)
    def setInputDevice(self, index: int) -> None:
        """Set the input device by ComboBox index (0 = System Default)."""

        ts_vs = vii.app.view_state.settings_modal_VS.transcription_section_vs
        if index <= 0:
            device_id = ""
        else:
            devices = ts_vs.available_input_devices
            if index - 1 < len(devices):
                device_id = devices[index - 1]["id"]
            else:
                device_id = ""

        set_input_device(device_id)

    @Slot(str)
    def transcribeFile(self, file_path: str) -> None:
        """Start transcription of the given audio file."""
        transcribe_file(file_path)

    @Slot()
    def openRecordingsFolder(self) -> None:
        """Open the recordings folder in the system file manager."""
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(RECORDINGS_DIR)))
