from __future__ import annotations

import asyncio
import json
import threading
from typing import TYPE_CHECKING, Optional

import websockets
from fusion.libs.entity.change import Change

from assistant.facade import facade
from assistant.inference.interface import parse_message

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from assistant.view_states.settings import AssistantSettingsViewState

CONNECT_TIMEOUT_SECONDS = 3.0


def _summarize_change(change: Change) -> str:
    try:
        item = change.new_state or change.old_state
        item_id = getattr(item, "id", None)
        content = getattr(item, "content", None) if item else None
        request = getattr(item, "request", None) if item else None
        content_keys = list(content.keys()) if isinstance(content, dict) else []
        request_keys = list(request.keys()) if isinstance(request, dict) else []
        origin = None
        if item and hasattr(item, "metadata"):
            meta = getattr(item, "metadata") or {}
            if isinstance(meta, dict):
                origin = meta.get("origin")
        return (
            f"type={change.change_type.name} id={item_id} "
            f"content_keys={content_keys} request_keys={request_keys} origin={origin}"
        )
    except Exception as exc:  # noqa: BLE001
        return f"type={getattr(change, 'change_type', '?')} (summary_failed: {exc})"


class ContextSyncClient:
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
        self._loop: asyncio.AbstractEventLoop | None = None
        try:
            self._settings_state: AssistantSettingsViewState | None = (
                facade.app_state.settings_VS
            )
        except Exception:
            self._settings_state = None
        self._connection_established = False
        self._last_error_message: Optional[str] = None
        self._pending_outbound: list[Change] = []

        # Subscribe to outbound client changes
        self._subscription = self._client_updates.subscribe(self._on_client_change)

    # --- Lifecycle -------------------------------------------------
    def start(self) -> None:
        thread = threading.Thread(target=self._run_loop, daemon=True)
        thread.start()

    def stop(self) -> None:
        self._stop = True
        if self._subscription:
            try:
                self._subscription.unsubscribe()
            except Exception:
                pass
            self._subscription = None
        try:
            if self._ws:
                asyncio.run(self._ws.close())
        except Exception:
            pass

    # --- Internal loop --------------------------------------------
    def _run_loop(self) -> None:  # pragma: no cover - network side effects
        try:
            asyncio.run(self._connect_and_pump())
        except Exception as exc:  # noqa: BLE001
            if not self._stop:
                print(f"InferenceClient: connection error {exc}")
                self._handle_connection_error(exc)

    async def _connect_and_pump(self):
        try:
            async with websockets.connect(
                self._url,
                open_timeout=CONNECT_TIMEOUT_SECONDS,
            ) as ws:
                self._loop = asyncio.get_running_loop()
                self._ws = ws
                self._handle_connected()
                print(f"InferenceClient: connected to {self._url}")
                try:
                    async for raw in ws:
                        self._handle_inbound_message(raw)
                except websockets.ConnectionClosed as exc:
                    print(
                        "InferenceClient: websocket closed"
                        f" code={exc.code} reason={exc.reason}"
                    )
                    self._handle_connection_closed()
                finally:
                    try:
                        await ws.close()
                    except Exception:
                        pass
        except Exception as exc:  # noqa: BLE001
            print(f"InferenceClient: connect/pump exception {exc}")
            self._handle_connection_error(exc)
        finally:
            self._ws = None
            self._loop = None
            self._connection_established = False

    def _handle_inbound_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            print(f"InferenceClient: failed to decode message {exc}")
            return

        try:
            inbound = parse_message(msg)
        except Exception as exc:  # noqa: BLE001
            print(f"InferenceClient: invalid message {exc} raw={msg}")
            return

        if inbound["type"] == "Change":
            self._handle_inbound_change(inbound)
        else:
            self._handle_inbound_append(inbound)

    def _handle_inbound_change(self, inbound: dict) -> None:
        try:
            change = Change.from_safe_delta_dict(inbound["payload"])
        except Exception as exc:  # noqa: BLE001
            print(f"InferenceClient: failed to parse change {exc} raw={inbound}")
            return
        print(f"InferenceClient: inbound change {_summarize_change(change)}")
        self._inference_updates.push(change)

    def _handle_inbound_append(self, inbound: dict) -> None:
        payload = inbound.get("payload", {})
        print(
            "InferenceClient: inbound append item_id=%s text_len=%d"
            % (payload.get("item_id"), len(payload.get("text", "")))
        )
        self._inference_updates.push(inbound)

    # --- Outbound handling ----------------------------------------
    def _on_client_change(self, change: Change) -> None:
        try:
            if not self._ws or getattr(self._ws, "closed", False):
                self._pending_outbound.append(change)
                return
            if not self._loop or self._loop.is_closed():
                self._pending_outbound.append(change)
                return
            self._send_change(change)
        except Exception as exc:  # noqa: BLE001
            print(f"InferenceClient: outbound send failed {exc}")

    def _send_change(self, change: Change) -> None:
        payload = {"type": "Change", "payload": change.as_safe_delta_dict()}
        print(f"InferenceClient: outbound change {_summarize_change(change)}")
        fut = asyncio.run_coroutine_threadsafe(
            self._ws.send(json.dumps(payload)),
            self._loop,
        )
        fut.add_done_callback(
            lambda f: (
                print(f"InferenceClient: outbound send error {f.exception()}")
                if f.exception()
                else None
            )
        )

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

        # Flush any changes buffered before the connection was ready
        pending = self._pending_outbound
        self._pending_outbound = []
        for change in pending:
            self._send_change(change)

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
