"""Thin wrapper around WebSocketSyncService (receiver role) for the desktop client.

Connects to the inference server's ``/ws/context`` endpoint, receives the
full context state, and keeps the local ContextStore in sync.  Outbound
changes from the local store are pushed automatically via
``sync.on_store_changes``.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import TYPE_CHECKING, Optional

import websockets
from fusion.logging import get_logger
from fusion.storage.delta import Delta
from fusion.storage.ws_sync_service import WebSocketSyncService

from assistant.inference.context_store import ContextStore

if TYPE_CHECKING:
    from assistant.view_states.settings import AssistantSettingsViewState

log = get_logger(__name__)

CONNECT_TIMEOUT_SECONDS = 3.0


class ContextSyncClient:
    """WebSocket sync client (receiver) for the desktop assistant."""

    def __init__(self, url: str, store: ContextStore) -> None:
        self._url = url
        self._store = store
        self._stop = False
        self._connection_established = False
        self._last_error_message: Optional[str] = None
        self._settings_state: Optional[AssistantSettingsViewState] = None
        self._ready = threading.Event()
        self.on_ready: Optional[callable] = None  # called once sync loop is running

        self._sync = WebSocketSyncService(store, role="receiver")

    def set_settings_state(self, state: AssistantSettingsViewState) -> None:
        self._settings_state = state

    def wait_ready(self, timeout: float = 5.0) -> bool:
        """Block until the sync handshake is done and the send loop is running."""
        return self._ready.wait(timeout)

    def _signal_ready(self) -> None:
        self._ready.set()
        if self.on_ready:
            try:
                self.on_ready()
            except Exception:
                pass

    # --- Lifecycle ---
    def start(self) -> None:
        thread = threading.Thread(target=self._run_loop, daemon=True)
        thread.start()

    def stop(self) -> None:
        self._stop = True

    # --- Internal loop ---
    def _run_loop(self) -> None:
        try:
            asyncio.run(self._connect_and_sync())
        except Exception as exc:  # noqa: BLE001
            if not self._stop:
                log.error("ContextSyncClient connection error: %s", exc)
                self._handle_connection_error(exc)

    async def _connect_and_sync(self) -> None:
        prev_on_changes = self._store.on_changes
        try:
            async with websockets.connect(
                self._url,
                open_timeout=CONNECT_TIMEOUT_SECONDS,
            ) as ws:
                self._handle_connected()

                # Chain store.on_changes: sync outbound + previous callback
                def _chained_on_changes(delta, origin=None):
                    self._sync.on_store_changes(delta, origin)
                    if prev_on_changes:
                        prev_on_changes(delta, origin)

                self._store.on_changes = _chained_on_changes

                async def send(msg: dict) -> None:
                    await ws.send(json.dumps(msg))

                # Wrap receive to detect handshake completion.
                # The receiver's first receive() returns the full_state message.
                # After sync.run() processes it, the concurrent send/receive
                # loops start. We signal ready after the first receive so that
                # callers waiting on wait_ready() can begin adding items.
                handshake_received = False

                async def receive_wrapper() -> dict:
                    nonlocal handshake_received
                    raw = await ws.recv()
                    if not handshake_received:
                        handshake_received = True
                        # Schedule signal after sync.run processes the full_state
                        asyncio.get_event_loop().call_soon(self._signal_ready)
                    return json.loads(raw)

                try:
                    await self._sync.run(send, receive_wrapper)
                except websockets.ConnectionClosed as exc:
                    log.info(
                        "ContextSyncClient: ws closed code=%s reason=%s",
                        exc.code,
                        exc.reason,
                    )
                    self._handle_connection_closed()
        except Exception as exc:  # noqa: BLE001
            log.error("ContextSyncClient connect/sync error: %s", exc)
            self._handle_connection_error(exc)
        finally:
            self._store.on_changes = prev_on_changes
            self._connection_established = False

    # --- Status callbacks ---
    def _handle_connection_error(self, exc: Exception) -> None:
        if isinstance(exc, TimeoutError):
            message = (
                f"Timed out after {CONNECT_TIMEOUT_SECONDS:.1f}s connecting to "
                f"{self._url}. Retrying…"
            )
        else:
            message = f"Could not connect to inference websocket at {self._url}: {exc}"
        if message == self._last_error_message:
            return
        self._last_error_message = message
        self._connection_established = False
        self._post_message(message)

    def _handle_connected(self) -> None:
        self._connection_established = True
        self._last_error_message = None
        self._post_message(f"Connected to inference websocket at {self._url}")

    def _handle_connection_closed(self) -> None:
        if not self._connection_established:
            return
        self._connection_established = False
        if self._stop:
            return
        self._post_message("Inference websocket connection closed")

    def _post_message(self, message: str) -> None:
        if not self._settings_state:
            return
        try:
            self._settings_state.post_info_message(message)
        except Exception:
            pass
