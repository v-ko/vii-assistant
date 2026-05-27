"""Actions for snippet overlay state management.

show_snippet_overlays: captures screenshots and creates view states per screen.
hide_snippet_overlays: removes all snippet overlay view states.
"""

from __future__ import annotations

from fusion.libs.action import action
from PySide6.QtGui import QGuiApplication

from assistant.facade import vii
from assistant.view_states.snippet import SnippetOverlayViewState


@action("snippet.show_overlays")
def show_snippet_overlays() -> None:
    """Capture screenshots and create a SnippetOverlayViewState per screen."""
    app_state = vii.app_state

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
def hide_snippet_overlays() -> None:
    """Remove all snippet overlay view states."""
    from assistant.util import get_logger

    log = get_logger(__name__)

    app_state = vii.app_state

    log.info(
        "hide_snippet_overlays: removing %d view states",
        len(app_state.snippet_overlays),
    )
    for vs in app_state.snippet_overlays:
        vs.setParent(None)
    app_state.snippet_overlays.clear()

    app_state.snippet_overlays_changed.emit()
