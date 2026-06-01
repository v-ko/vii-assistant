"""Actions for snippet overlay state management.

show_snippet_overlays: captures screenshots and creates view states per screen.
hide_snippet_overlays: removes all snippet overlay view states.
"""

from __future__ import annotations

from fusion.libs.action import action
from PySide6.QtGui import QGuiApplication

from assistant.util import get_logger
from assistant.view_states.snippet import SnippetOverlayViewState

log = get_logger(__name__)


@action("snippet.show_overlays")
def show_snippet_overlays(app_state) -> None:
    """Capture screenshots and create a SnippetOverlayViewState per screen."""
    # If already showing, do nothing
    if app_state.snippet_overlays:
        return

    screens = QGuiApplication.screens()
    for screen in screens:
        pixmap = screen.grabWindow(0)
        screenshot = pixmap.toImage()
        vs = SnippetOverlayViewState(
            screen_name=screen.name(),
            geometry=screen.geometry(),
            screenshot=screenshot,
            parent=app_state,
        )
        app_state.snippet_overlays.append(vs)

    app_state.snippet_overlays_changed.emit()


@action("snippet.hide_overlays")
def hide_snippet_overlays(app_state) -> None:
    """Remove all snippet overlay view states."""
    for vs in app_state.snippet_overlays:
        vs.setParent(None)
    app_state.snippet_overlays.clear()

    app_state.snippet_overlays_changed.emit()
