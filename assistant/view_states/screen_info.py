"""View state for a single screen's info, exposed to QML."""

from __future__ import annotations

from dataclasses import dataclass

from fusion.platform.qt_widgets import Property
from PySide6.QtCore import QObject, Signal


@dataclass
class ScreenInfoData:
    """Plain data transfer object for screen info (input to projector)."""

    name: str
    x: int
    y: int
    width: int
    height: int
    is_primary: bool
    is_capture: bool


class ScreenInfoVS(QObject):
    """QML-bindable view state for a single screen."""

    changed = Signal()

    def __init__(self, data: ScreenInfoData, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._name = data.name
        self._x = data.x
        self._y = data.y
        self._width = data.width
        self._height = data.height
        self._is_primary = data.is_primary
        self._is_capture = data.is_capture

    def update_from(self, data: ScreenInfoData) -> bool:
        """Update fields from data. Returns True if anything changed."""
        changed = (
            self._name != data.name
            or self._x != data.x
            or self._y != data.y
            or self._width != data.width
            or self._height != data.height
            or self._is_primary != data.is_primary
            or self._is_capture != data.is_capture
        )
        if not changed:
            return False
        self._name = data.name
        self._x = data.x
        self._y = data.y
        self._width = data.width
        self._height = data.height
        self._is_primary = data.is_primary
        self._is_capture = data.is_capture
        self.changed.emit()
        return True

    @Property(str, notify=changed)
    def name(self) -> str:
        return self._name

    @Property(int, notify=changed)
    def x(self) -> int:
        return self._x

    @Property(int, notify=changed)
    def y(self) -> int:
        return self._y

    @Property(int, notify=changed)
    def width(self) -> int:
        return self._width

    @Property(int, notify=changed)
    def height(self) -> int:
        return self._height

    @Property(bool, notify=changed)
    def is_primary(self) -> bool:
        return self._is_primary

    @Property(bool, notify=changed)
    def is_capture(self) -> bool:
        return self._is_capture
