from PySide6.QtCore import QObject, QRect, Signal
from PySide6.QtGui import QImage


class SnippetOverlayViewState(QObject):
    """View state for a single snippet overlay (one per screen).

    Created by the show_snippet_overlays action with a captured screenshot
    and screen geometry. Destroyed by hide_snippet_overlays or when screen
    configuration changes.
    """

    def __init__(
        self,
        screen_name: str,
        geometry: QRect,
        screenshot: QImage,
        parent=None,
    ):
        super().__init__(parent)
        self.screen_name = screen_name
        self.geometry = geometry
        self.screenshot = screenshot
