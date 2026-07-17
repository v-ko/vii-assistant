"""Write-authority guard for the shared context store.

The context store is written by two processes that share one synced copy:

- ``"vii-assistant"`` — the desktop client / GUI.
- ``"inference-server"`` — the model server (possibly on another host).

Most fields are owned by exactly one side. This guard is a *development-time
assertion*: it fires on every local write and raises ``ValueError`` when a side
touches a field it does not own, so the offending code path fails loud instead
of silently producing a split-brain state.

It is wired as the FIRST on-changes callback on each store, with the role
hard-coded via a lambda::

    store.add_on_changes_callback(lambda d, o: authority_guard("inference-server", d, o))
    store.add_on_changes_callback(lambda d, o: authority_guard("vii-assistant", d, o))

Remote deltas were already validated at their source store, so they are skipped.
"""

from __future__ import annotations

from sivkit.storage.delta import Delta


def authority_guard(role: str, delta: Delta, origin: str | None) -> None:
    """Raise ValueError if ``role`` writes a field it does not own.

    Only local writes are checked; remote deltas (``origin == "remote"``) were
    already validated at the peer that produced them.
    """
    if origin == "remote":
        return
    for change in delta.changes():
        if change.is_create():
            # Items are created with their full state by whichever side births
            # them; ownership only constrains subsequent updates.
            continue
        fields = change.forward_component or {}
        request = fields.get("request")
        req = request if isinstance(request, dict) else {}
        if role == "inference-server":
            if "cancelled" in fields:
                raise ValueError("inference-server may not write 'cancelled'")
            tf = fields.get("teacher_feedback")
            if tf is not None and tf != "pending":
                raise ValueError("inference-server may not write a verdict")
        elif role == "vii-assistant":
            if "result" in req or "completed" in req:
                raise ValueError("vii-assistant may not write request.result/completed")
