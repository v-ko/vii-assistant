"""OverlayManager — manages primary overlay + per-screen guard overlays.

The primary overlay (on the assistant's target screen) shows model shapes,
confirmation prompts, etc. Guard overlays appear on other screens in AUTO
mode to signal the user that the assistant is working.
"""

from __future__ import annotations

import logging

from PySide6.QtGui import QGuiApplication

from assistant.view_states.overlay import OverlayMode, OverlayViewState
from assistant.widgets.overlay import ModelVisionOverlay

log = logging.getLogger(__name__)


class OverlayManager:
    """Creates/destroys overlays based on execution mode and assistant activity."""

    def __init__(self) -> None:
        self._guard_overlays: dict[str, tuple[OverlayViewState, ModelVisionOverlay]] = (
            {}
        )

    def activate_auto_guards(self, exclude_screen: str) -> None:
        """Show green guard overlays on all screens except the one being acted on."""
        screens = QGuiApplication.screens()
        for screen in screens:
            name = screen.name()
            if name == exclude_screen:
                continue
            if name in self._guard_overlays:
                continue
            state = OverlayViewState()
            state.mode = OverlayMode.AUTO_GUARD
            state.screen_name = name
            overlay = ModelVisionOverlay(state)
            overlay.show()
            self._guard_overlays[name] = (state, overlay)
            log.info("Guard overlay activated on screen '%s'", name)

    def deactivate_guards(self) -> None:
        """Remove all guard overlays."""
        for name, (state, overlay) in self._guard_overlays.items():
            overlay.hide()
            overlay.deleteLater()
            log.info("Guard overlay deactivated on screen '%s'", name)
        self._guard_overlays.clear()

    @property
    def guards_active(self) -> bool:
        return bool(self._guard_overlays)
