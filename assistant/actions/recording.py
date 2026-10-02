"""Actions for recording overlay state management.

All recording overlay state mutations go through @action-decorated functions.
The view model reacts to view state signals automatically.
"""

from __future__ import annotations

from sivkit.libs.action import action

from assistant.facade import vii
from assistant.view_states.recording_overlay import RecordingOverlayViewState


@action("recording.set_overlay_visible")
def set_overlay_visible(visible: bool) -> None:
    """Show or hide the recording overlay view state."""
    if visible:
        if vii.app.view_state.recording_overlay_VS is None:
            vii.app.view_state.recording_overlay_VS = RecordingOverlayViewState(
                parent=vii.app.view_state
            )
    else:
        if vii.app.view_state.recording_overlay_VS is not None:
            vii.app.view_state.recording_overlay_VS = None


def _maybe_hide(vs: RecordingOverlayViewState) -> None:
    """Hide overlay when both modes are inactive."""
    if not vs.recording_active and not vs.transcribing_active:
        set_overlay_visible(False)


@action("recording.set_recording_active")
def set_recording_active(active: bool) -> None:
    """Update recording_active flag on the overlay view state."""
    vs = vii.app.view_state.recording_overlay_VS
    if vs is None:
        if active:
            set_overlay_visible(True)
            vs = vii.app.view_state.recording_overlay_VS
            assert vs is not None
        else:
            return
    vs.recording_active = active
    _maybe_hide(vs)


@action("recording.set_transcribing_active")
def set_transcribing_active(active: bool) -> None:
    """Update transcribing_active flag on the overlay view state."""
    vs = vii.app.view_state.recording_overlay_VS
    if vs is None:
        if active:
            set_overlay_visible(True)
            vs = vii.app.view_state.recording_overlay_VS
            assert vs is not None
        else:
            return
    vs.transcribing_active = active
    _maybe_hide(vs)
