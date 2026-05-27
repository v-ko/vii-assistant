from __future__ import annotations

from enum import Enum

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage

from assistant.util import Shape


class OverlayMode(Enum):
    WORK = "work"
    EXPERIMENT = "experiment"
    AUTO_GUARD = "auto-guard"
    CONFIRM = "confirm"


class OverlayViewState(QObject):
    mode_changed = Signal(OverlayMode)
    shapes_changed = Signal(list)
    gt_shapes_changed = Signal(list)
    sample_image_changed = Signal(object)  # QImage | None
    screen_name_changed = Signal(str)
    dimmed_changed = Signal(bool)
    pending_actions_changed = Signal(list)  # list[str] — action descriptions

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._mode = OverlayMode.WORK
        self._shapes: list[Shape] = []
        self._gt_shapes: list[Shape] = []
        self._sample_image: QImage | None = None
        self._screen_name: str = ""
        self._dimmed: bool = False
        self._pending_actions: list[str] = []

    @property
    def mode(self) -> OverlayMode:
        return self._mode

    @mode.setter
    def mode(self, value: OverlayMode) -> None:
        if self._mode == value:
            return
        self._mode = value
        self.mode_changed.emit(value)

    @property
    def shapes(self) -> list[Shape]:
        return self._shapes

    @shapes.setter
    def shapes(self, value: list[Shape]) -> None:
        self._shapes = value
        self.shapes_changed.emit(value)

    @property
    def gt_shapes(self) -> list[Shape]:
        return self._gt_shapes

    @gt_shapes.setter
    def gt_shapes(self, value: list[Shape]) -> None:
        self._gt_shapes = value
        self.gt_shapes_changed.emit(value)

    @property
    def sample_image(self) -> QImage | None:
        return self._sample_image

    @sample_image.setter
    def sample_image(self, value: QImage | None) -> None:
        self._sample_image = value
        self.sample_image_changed.emit(value)

    @property
    def screen_name(self) -> str:
        return self._screen_name

    @screen_name.setter
    def screen_name(self, value: str) -> None:
        if self._screen_name == value:
            return
        self._screen_name = value
        self.screen_name_changed.emit(value)

    @property
    def dimmed(self) -> bool:
        return self._dimmed

    @dimmed.setter
    def dimmed(self, value: bool) -> None:
        if self._dimmed == value:
            return
        self._dimmed = value
        self.dimmed_changed.emit(value)

    @property
    def pending_actions(self) -> list[str]:
        return self._pending_actions

    @pending_actions.setter
    def pending_actions(self, value: list[str]) -> None:
        self._pending_actions = value
        self.pending_actions_changed.emit(value)

    def clear(self) -> None:
        self.shapes = []
        self.gt_shapes = []
        self.sample_image = None
        self.dimmed = False
        self.pending_actions = []
        self.mode = OverlayMode.WORK
