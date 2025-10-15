from __future__ import annotations

import json
import threading
import time
from typing import TYPE_CHECKING, Optional

import websockets
from fusion.libs.entity.change import Change

from assistant.facade import facade
from assistant.inference.interface import (
    AppendItemContentTextMessage,
    ChangeMessage,
    parse_message,
)

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from assistant.view_states.settings import SettingsViewState

CONNECT_TIMEOUT_SECONDS = 3.0
RECONNECT_DELAY_SECONDS = 3.0


class InferenceClient:
    """Minimal websocket client syncing context changes with server.

    Outbound: pushes Change objects published to client_updates Channel.
    Inbound: receives wrapped Change / streaming fragments and pushes onto
    inference_updates Channel for AutomationService to apply.
    """

    def __init__(
        self,
        url: str,
    ) -> None:
        self._url = url
        self._client_updates = facade.client_updates
        self._inference_updates = facade.inference_updates
        self._stop = False
        self._ws = None
        try:
            self._settings_state: SettingsViewState | None = (
                facade.app_state.settings_VS
            )
        except Exception:
            self._settings_state = None
        self._connection_established = False
        self._last_error_message: Optional[str] = None

        # Subscribe to outbound client changes
        self._client_updates.subscribe(self._on_client_change)

    # --- Lifecycle -------------------------------------------------
    def start(self) -> None:
        thread = threading.Thread(target=self._run_loop, daemon=True)
        thread.start()

    def stop(self) -> None:  # pragma: no cover - best effort
        self._stop = True
        try:
            if self._ws:
                import asyncio

                asyncio.run(self._ws.close())
        except Exception:
            pass

    # --- Internal loop --------------------------------------------
    def _run_loop(self) -> None:  # pragma: no cover - network side effects
        while not self._stop:
            try:
                import asyncio

                asyncio.run(self._connect_and_pump())
            except Exception as exc:  # noqa: BLE001
                if not self._stop:
                    print(f"InferenceClient: connection error {exc}")
                    self._handle_connection_error(exc)
            if self._stop:
                break
            time.sleep(RECONNECT_DELAY_SECONDS)

    async def _connect_and_pump(self):  # noqa: C901 - acceptable complexity
        async for ws in websockets.connect(
            self._url,
            open_timeout=CONNECT_TIMEOUT_SECONDS,
        ):  # auto-reconnect pattern
            self._ws = ws
            self._handle_connected()
            print(f"InferenceClient: connected to {self._url}")
            try:
                # Receive loop
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue
                    try:
                        inbound = parse_message(msg)
                    except Exception:
                        continue
                    if inbound["type"] == "Change":
                        try:
                            change = Change.from_safe_delta_dict(
                                inbound["payload"]
                            )  # strict dict
                        except Exception:
                            continue
                        self._inference_updates.push(change)
                    else:  # AppendItemContentTextMessage
                        self._inference_updates.push(inbound)
            except websockets.ConnectionClosed:  # pragma: no cover
                print("InferenceClient: websocket closed; reconnecting…")
                self._handle_connection_closed()
                continue
            finally:
                try:
                    await ws.close()
                except Exception:
                    pass
                self._ws = None
                if not self._stop:
                    self._connection_established = False

    # --- Outbound handling ----------------------------------------
    def _on_client_change(self, change: Change) -> None:
        # Send Change as wrapped structure; best effort non-blocking
        try:
            import asyncio

            if not self._ws:  # not connected yet
                return
            if getattr(self._ws, "closed", False):
                return
            payload = {"type": "Change", "payload": change.as_safe_delta_dict()}
            asyncio.run(self._ws.send(json.dumps(payload)))
        except Exception:
            pass

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
        if self._connection_established:
            return
        self._connection_established = True
        self._last_error_message = None
        self._post_message(f"Connected to inference websocket at {self._url}")

    def _handle_connection_closed(self) -> None:
        if not self._connection_established:
            return
        self._connection_established = False
        if self._stop:
            return
        self._post_message("Inference websocket connection closed; retrying")

    def _post_message(self, message: str) -> None:
        if not self._settings_state:
            return
        try:
            self._settings_state.post_info_message(message)
        except Exception:
            pass
