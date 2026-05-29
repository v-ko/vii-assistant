"""Screen layout projector.

Maps system screen state (compiled by the app class) into ScreenInfoVS
objects on app_state. Also handles major view state operations triggered
by screen changes (e.g. cancelling stale snippet overlays).
"""

from __future__ import annotations

from fusion.libs.action import action

from assistant.facade import vii
from assistant.snippet_actions import hide_snippet_overlays
from assistant.util import get_logger
from assistant.view_states.screen_info import ScreenInfoData, ScreenInfoVS

log = get_logger(__name__)


@action("screen.project_layout", issuer="service")
def project_screen_layout(screen_data: list[ScreenInfoData]) -> None:
    """Update app_state.screens from compiled screen data.

    Diffs against existing list: updates in place where possible,
    adds/removes as needed. Emits screens_changed when done.

    Also cancels stale snippet overlays when screen list changes.
    """
    app_state = vii.app_state
    existing = app_state.screens
    existing_by_name = {s.name: s for s in existing}
    incoming_names = {d.name for d in screen_data}

    new_list: list[ScreenInfoVS] = []
    changed = False

    for data in screen_data:
        if data.name in existing_by_name:
            vs = existing_by_name[data.name]
            if vs.update_from(data):
                changed = True
            new_list.append(vs)
        else:
            # New screen appeared
            vs = ScreenInfoVS(data, parent=app_state)
            new_list.append(vs)
            changed = True

    # Detect removed screens
    if set(existing_by_name.keys()) != incoming_names:
        changed = True

    if not changed and len(new_list) == len(existing):
        return

    app_state.screens = new_list

    # Keep overlay_VS.screen_name in sync (ModelVisionOverlay reads it)
    capture = app_state.capture_screen_info
    if capture and app_state.overlay_VS.screen_name != capture.name:
        app_state.overlay_VS.screen_name = capture.name

    # Cancel active snippet overlays (screenshots are stale)
    if app_state.snippet_overlays:
        log.info("Screen layout changed — cancelling active snippet overlays")
        hide_snippet_overlays()
