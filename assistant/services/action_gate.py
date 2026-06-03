"""ActionGate — blocks tool execution until user confirms (USER_APPROVE mode).

In AUTO mode the gate is always open; tools execute immediately.
In USER_APPROVE mode the gate blocks until the user sends a confirm or stop command.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from sivkit.logging import get_logger

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
        self._future: asyncio.Future[bool] | None = None
        self._pending: list[PendingAction] = []

    @property
    def pending_actions(self) -> list[PendingAction]:
        return self._pending

    @property
    def is_waiting(self) -> bool:
        """True if the gate is currently blocking on user input."""
        return self._future is not None and not self._future.done()

    async def await_confirmation(self, actions: list[PendingAction]) -> GateResult:
        """Block until user confirms or interrupts.

        The caller should update the overlay to show pending actions before calling.
        If a previous waiter is still pending it is interrupted first.
        """
        # Cancel any stale waiter from a previous session/request
        if self._future is not None and not self._future.done():
            log.info("ActionGate: cancelling stale waiter")
            self._future.set_result(False)

        self._pending = actions
        loop = asyncio.get_running_loop()
        self._future = loop.create_future()
        log.info("ActionGate: waiting for user confirmation (%d actions)", len(actions))
        confirmed = await self._future
        self._pending = []
        return GateResult(confirmed=confirmed)

    def confirm(self) -> None:
        """Unblock the gate — user approved the pending actions."""
        if not self.is_waiting:
            log.warning("ActionGate.confirm() called but gate is not waiting")
            return
        log.info("ActionGate: user confirmed")
        self._future.set_result(True)

    def interrupt(self) -> None:
        """Unblock the gate — user rejected / stopped."""
        log.info("ActionGate: user interrupted")
        if self._future is not None and not self._future.done():
            self._future.set_result(False)
        # Also clear pending so overlay can react immediately
        self._pending = []
