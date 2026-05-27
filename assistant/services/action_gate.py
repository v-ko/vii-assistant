"""ActionGate — blocks tool execution until user confirms (USER_APPROVE mode).

In AUTO mode the gate is always open; tools execute immediately.
In USER_APPROVE mode the gate blocks until the user sends a confirm or stop command.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from fusion.logging import get_logger

log = get_logger(__name__)


@dataclass
class PendingAction:
    """A tool call awaiting user confirmation."""

    call_id: str
    tool_name: str
    description: str


@dataclass
class GateResult:
    """Result of awaiting the gate."""

    confirmed: bool
    """True if user confirmed, False if interrupted."""


class ActionGate:
    """Blocks tool execution in USER_APPROVE mode until user confirms."""

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._interrupted: bool = False
        self._pending: list[PendingAction] = []

    @property
    def pending_actions(self) -> list[PendingAction]:
        return self._pending

    @property
    def is_waiting(self) -> bool:
        """True if the gate is currently blocking on user input."""
        return bool(self._pending) and not self._event.is_set()

    async def await_confirmation(self, actions: list[PendingAction]) -> GateResult:
        """Block until user confirms or interrupts.

        The caller should update the overlay to show pending actions before calling.
        """
        self._pending = actions
        self._interrupted = False
        self._event.clear()
        log.info("ActionGate: waiting for user confirmation (%d actions)", len(actions))
        await self._event.wait()
        self._pending = []
        return GateResult(confirmed=not self._interrupted)

    def confirm(self) -> None:
        """Unblock the gate — user approved the pending actions."""
        if not self.is_waiting:
            log.warning("ActionGate.confirm() called but gate is not waiting")
            return
        log.info("ActionGate: user confirmed")
        self._interrupted = False
        self._event.set()

    def interrupt(self) -> None:
        """Unblock the gate — user rejected / stopped."""
        log.info("ActionGate: user interrupted")
        self._interrupted = True
        self._event.set()
        # Also clear pending so overlay can react immediately
        self._pending = []
