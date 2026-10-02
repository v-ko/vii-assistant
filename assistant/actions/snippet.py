"""Actions for snippet overlay state management.

show_snippet_overlays: creates view states per screen from captured screenshots.
hide_snippet_overlays: removes all snippet overlay view states.
"""

from __future__ import annotations

import logging

from PySide6.QtGui import QGuiApplication, QPixmap
from sivkit.libs.action import action

from assistant.components.snippet_overlay.view_state import SnippetOverlayViewState

log = logging.getLogger(__name__)


@action("snippet.show_overlays")
def show_snippet_overlays(app_state, screenshots: dict[str, QPixmap]) -> None:
    """Create a SnippetOverlayViewState per captured screen."""
    # If already showing, do nothing
    if app_state.snippet_overlays:
        return

    screens = QGuiApplication.screens()
    for screen in screens:
        pixmap = screenshots.get(screen.name())
        if pixmap is None or pixmap.isNull():
            log.warning("No screenshot for screen '%s', skipping", screen.name())
            continue
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
