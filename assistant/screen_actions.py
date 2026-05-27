"""Action for reacting to screen configuration changes.

Called whenever screens are added, removed, or change geometry.
This is the single place where all view state adjustments for screen
changes are made.
"""

from __future__ import annotations

from fusion.libs.action import action

from assistant.facade import vii
from assistant.snippet_actions import hide_snippet_overlays
from assistant.util import get_logger, get_screen_by_name

log = get_logger(__name__)


@action("screen.apply_config_change")
def apply_screen_config_change() -> None:
    """Adapt view states to a changed screen configuration.

    - Revalidate the watched screen (fallback if removed)
    - Update terminal view state geometry
    - Update the main overlay view state
    - Cancel any active snippet overlays (stale screenshots)
    """
    app_state = vii.app_state
    settings = app_state.settings_VS

    # --- Validate watched screen ---
    current_screen_name = settings.screen
    screen = get_screen_by_name(current_screen_name) if current_screen_name else None

    if screen is None:
        fallback = vii.get_default_screen()
        log.info(
            "Watched screen '%s' gone, falling back to '%s'",
            current_screen_name,
            fallback.name(),
        )
        settings.screen = fallback.name()
        screen = fallback

    # --- Update terminal geometry view state ---
    geo = screen.geometry()
    terminal_w = int(geo.width() * 0.9)
    terminal_h = geo.height() // 2
    terminal_x = geo.x() + (geo.width() - terminal_w) // 2
    terminal_y = geo.y()

    qt_app = vii.qt_app
    qt_app.terminal_state.set_geometry(terminal_x, terminal_y, terminal_w, terminal_h)

    # --- Update main overlay view state ---
    app_state.overlay_VS.screen_name = screen.name()

    # --- Cancel active snippet overlays (screenshots are stale) ---
    if app_state.snippet_overlays:
        log.info("Screen config changed — cancelling active snippet overlays")
        hide_snippet_overlays()
