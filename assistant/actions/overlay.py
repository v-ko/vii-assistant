from __future__ import annotations

from sivkit.libs.action import action

from assistant.facade import vii
from assistant.view_states.overlay import OverlayMode


@action("overlay.show_pending_actions")
def show_pending_actions(descriptions: list[str]) -> None:
    overlay_vs = vii.app.view_state.overlay_VS
    overlay_vs.pending_actions = descriptions
    overlay_vs.mode = OverlayMode.CONFIRM


@action("overlay.clear_pending_actions")
def clear_pending_actions() -> None:
    overlay_vs = vii.app.view_state.overlay_VS
    overlay_vs.pending_actions = []
    overlay_vs.mode = OverlayMode.WORK


@action("overlay.show_supervised_review")
def show_supervised_review(text: str) -> None:
    overlay_vs = vii.app.view_state.overlay_VS
    overlay_vs.review_text = text
    overlay_vs.mode = OverlayMode.SUPERVISED_REVIEW


@action("overlay.clear_supervised_review")
def clear_supervised_review() -> None:
    overlay_vs = vii.app.view_state.overlay_VS
    overlay_vs.review_text = ""
    overlay_vs.mode = OverlayMode.WORK
