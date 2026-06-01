"""Debug actions — state mutations for debug/development tools."""

from __future__ import annotations

from fusion.libs.action import action

from assistant.facade import vii


@action("debug.show_screen_debug")
def show_screen_debug() -> None:
    """Show the screen layout debug window."""
    vii.app.view_state.screen_debug_visible = True


@action("debug.hide_screen_debug")
def hide_screen_debug() -> None:
    """Hide the screen layout debug window."""
    vii.app.view_state.screen_debug_visible = False
