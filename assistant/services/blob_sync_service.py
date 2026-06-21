"""WebSocket sync service extended with blob transport.

Subclasses sivkit's WebSocketSyncService to additionally sync blobs
from a BlobStore over the same WebSocket connection. Blobs are sent
automatically on creation (via BlobStore.on_added callback) and received
blobs are written to the local BlobStore.

Wire format for blob messages::

    {"type": "blob", "ref": "ab/abcdef1234.png", "data": "<base64>"}

Blob messages are interleaved with normal entity delta messages on the
same WebSocket. WS ordering guarantees that a blob arrives before the
entity delta referencing it (caller must add blob before inserting entity).
"""

from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from sivkit.storage.base_store import Store
from sivkit.storage.blob_store import BlobStore
from sivkit.storage.websocket_sync_service import (
    ReceiveFn,
    SendFn,
    SyncMode,
    WebSocketSyncService,
)

log = logging.getLogger(__name__)


class BlobAwareSyncService(WebSocketSyncService):
    """WebSocketSyncService with blob auto-sync over the same connection."""

    def __init__(
        self,
        store: Store,
        blob_store: BlobStore,
        role: str,
        mode: SyncMode = SyncMode.OT,
        on_ready: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(store, role=role, mode=mode, on_ready=on_ready)
        self._blob_store = blob_store
        self._blob_outbound: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def run(self, send: SendFn, receive: ReceiveFn) -> None:
        """Run sync with blob support.

        Wraps the receive callable to intercept blob messages before they
        reach the base class, and wires the BlobStore callback for outbound.
        """
        loop = asyncio.get_running_loop()

        def _on_blob_added(ref: str, data: bytes) -> None:
            """Called synchronously from BlobStore.add — enqueue for async send."""
            if not self.running:
                return
            msg = {
                "type": "blob",
                "ref": ref,
                "data": base64.b64encode(data).decode("ascii"),
            }
            loop.call_soon_threadsafe(self._blob_outbound.put_nowait, msg)

        self._blob_store.add_on_added_callback(_on_blob_added)

        # Wrap send to also drain blob queue
        original_send = send

        async def _send_with_blobs(msg: dict[str, Any]) -> None:
            # Drain pending blobs before each entity message
            while not self._blob_outbound.empty():
                blob_msg = self._blob_outbound.get_nowait()
                await original_send(blob_msg)
            await original_send(msg)

        # Wrap receive to intercept blob messages
        async def _receive_filtering_blobs() -> dict[str, Any]:
            while True:
                msg = await receive()
                if msg.get("type") == "blob":
                    self._handle_incoming_blob(msg)
                    continue
                return msg

        try:
            await super().run(_send_with_blobs, _receive_filtering_blobs)
        finally:
            self._blob_store.remove_on_added_callback(_on_blob_added)
            # Drain any remaining blobs
            while not self._blob_outbound.empty():
                self._blob_outbound.get_nowait()

    def _handle_incoming_blob(self, msg: dict[str, Any]) -> None:
        """Write a received blob to local BlobStore."""
        ref = msg.get("ref", "")
        data_b64 = msg.get("data", "")
        if not ref or not data_b64:
            log.warning("Received malformed blob message: missing ref or data")
            return
        try:
            data = base64.b64decode(data_b64)
        except Exception:
            log.warning("Received blob with invalid base64: ref=%s", ref)
            return
        # BlobStore.add deduplicates — safe to call even if we already have it
        self._blob_store.add(data, extension=_extension_from_ref(ref))
        log.debug("Received and stored blob: %s (%d bytes)", ref, len(data))


def _extension_from_ref(ref: str) -> str:
    """Extract file extension from a blob ref like 'ab/abcdef1234.png'."""
    if "." in ref:
        return ref.rsplit(".", 1)[1]
    return ""
