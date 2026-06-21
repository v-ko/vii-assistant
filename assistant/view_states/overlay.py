from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage

from assistant.util import Shape

if TYPE_CHECKING:
    from assistant.image_ops import ResizeMetadata


@dataclass
class DisplayTransform:
    """Maps from original image pixel space to overlay widget pixel space.

    widget_x = image_x * scale + offset_x
    widget_y = image_y * scale + offset_y
    """

    scale: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    # Original image dimensions (for clamping/reference)
    image_width: int = 0
    image_height: int = 0

    def image_to_widget(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale + self.offset_x, y * self.scale + self.offset_y

    def image_rect_to_widget(
        self, x: float, y: float, w: float, h: float
    ) -> tuple[int, int, int, int]:
        wx, wy = self.image_to_widget(x, y)
        return int(wx), int(wy), int(w * self.scale), int(h * self.scale)


class OverlayMode(Enum):
    WORK = "work"
    EXPERIMENT = "experiment"
    AUTO_GUARD = "auto-guard"
    CONFIRM = "confirm"
    SUPERVISED_REVIEW = "supervised-review"


class OverlayViewState(QObject):
    mode_changed = Signal(OverlayMode)
    shapes_changed = Signal(list)
    gt_shapes_changed = Signal(list)
    sample_image_changed = Signal(object)  # QImage | None
    dimmed_changed = Signal(bool)
    pending_actions_changed = Signal(list)  # list[str] — action descriptions
    display_transform_changed = Signal(object)  # DisplayTransform
    progress_changed = Signal(int, int)  # current_step, total
    review_text_changed = Signal(str)  # agent response text for review

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._mode = OverlayMode.WORK
        self._shapes: list[Shape] = []
        self._gt_shapes: list[Shape] = []
        self._sample_image: QImage | None = None
        self._dimmed: bool = False
        self._pending_actions: list[str] = []
        self._display_transform = DisplayTransform()
        self._resize_meta: ResizeMetadata | None = None
        self._review_text: str = ""

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

    @property
    def display_transform(self) -> DisplayTransform:
        return self._display_transform

    @display_transform.setter
    def display_transform(self, value: DisplayTransform) -> None:
        self._display_transform = value
        self.display_transform_changed.emit(value)

    @property
    def resize_meta(self) -> ResizeMetadata | None:
        """ResizeMetadata dict from the model's input preprocessing."""
        return self._resize_meta

    @resize_meta.setter
    def resize_meta(self, value: ResizeMetadata | None) -> None:
        self._resize_meta = value

    @property
    def review_text(self) -> str:
        return self._review_text

    @review_text.setter
    def review_text(self, value: str) -> None:
        if self._review_text == value:
            return
        self._review_text = value
        self.review_text_changed.emit(value)

    def clear(self) -> None:
        self.shapes = []
        self.gt_shapes = []
        self.sample_image = None
        self.dimmed = False
        self.pending_actions = []
        self.display_transform = DisplayTransform()
        self.resize_meta = None
        self.review_text = ""
        self.mode = OverlayMode.WORK
