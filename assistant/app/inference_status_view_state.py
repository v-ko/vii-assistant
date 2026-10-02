from __future__ import annotations

from PySide6.QtCore import QObject, Signal
from sivkit.platform.qt_widgets import Property


class InferenceStatusViewState(QObject):
    """View state mirroring the inference server status.

    Fed by ``project_inference_status`` from the inference client's live
    attributes. The view binds to these properties and never reads the
    client directly.
    """

    connected_changed = Signal(bool)
    model_state_changed = Signal(str)  # "unknown" | "unloaded" | "loading" | "loaded"
    model_key_changed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._connected = False
        self._model_state = "unknown"
        self._model_key = ""

    @Property(bool, notify=connected_changed)
    def connected(self) -> bool:
        return self._connected

    @connected.setter
    def connected(self, value: bool) -> None:
        if self._connected == value:
            return
        self._connected = value
        self.connected_changed.emit(value)

    @Property(str, notify=model_state_changed)
    def model_state(self) -> str:
        return self._model_state

    @model_state.setter
    def model_state(self, value: str) -> None:
        if self._model_state == value:
            return
        self._model_state = value
        self.model_state_changed.emit(value)

    @Property(str, notify=model_key_changed)
    def model_key(self) -> str:
        return self._model_key

    @model_key.setter
    def model_key(self, value: str) -> None:
        if self._model_key == value:
            return
        self._model_key = value
        self.model_key_changed.emit(value)
