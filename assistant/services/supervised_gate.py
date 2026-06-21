"""SupervisedGate — tracks the turn currently awaiting a supervisor verdict.

Supervision is a synced-field gate: a completed turn carries
``teacher_feedback="pending"`` and the server withholds its continuation until
the client writes a verdict (pass / correct / error) onto the field. This gate
is the thin client-side state that remembers which turn an open review window
refers to, and routes the supervisor's button presses to the store write.

The actual verdict write + FeedbackItem insertion lives on
``HybridSegmentService.submit_supervised_verdict``; this object just holds the
pending item id and forwards.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class SupervisedGate:
    """Remembers the turn an open supervision window is reviewing."""

    def __init__(self) -> None:
        self._pending_item_id: str | None = None

    @property
    def pending_item_id(self) -> str | None:
        return self._pending_item_id

    @property
    def is_waiting(self) -> bool:
        return self._pending_item_id is not None

    def arm(self, item_id: str) -> None:
        """Mark a turn as awaiting a supervisor verdict (window shown)."""
        self._pending_item_id = item_id

    def clear(self) -> None:
        """Forget the pending turn (verdict submitted, or chain stopped)."""
        self._pending_item_id = None

    def _service(self):
        from assistant.facade import vii

        return vii.project_manager.hybrid_segment_service

    def submit_correct(self) -> None:
        self._submit("correct")

    def submit_pass(self) -> None:
        self._submit("pass")

    def submit_error(self, correction_text: str) -> None:
        self._submit("error", correction_text)

    def _submit(self, verdict: str, correction_text: str = "") -> None:
        item_id = self._pending_item_id
        if item_id is None:
            log.warning("SupervisedGate.submit %s but nothing pending", verdict)
            return
        log.info("SupervisedGate: verdict=%s item=%s", verdict, item_id)
        self._service().submit_supervised_verdict(item_id, verdict, correction_text)
