from __future__ import annotations

import json
import threading
import time
from typing import Optional

import websockets
from fusion.libs.entity.change import Change

from assistant.inference.interface import unwrap_message


class InferenceClient:
    """Minimal websocket client syncing context changes with server.

    Outbound: pushes Change objects published to client_updates Channel.
    Inbound: receives wrapped Change / streaming fragments and pushes onto
    inference_updates Channel for AutomationService to apply.
    """

    def __init__(
        self,
        *,
        client_updates,
        inference_updates,
        url: str = "ws://127.0.0.1:8010/ws/context",
        reconnect_seconds: float = 3.0,
    ) -> None:
        self._url = url
        self._client_updates = client_updates
        self._inference_updates = inference_updates
        self._reconnect_seconds = reconnect_seconds
        self._stop = False
        self._ws = None

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
                print(f"InferenceClient: connection error {exc}")
            if self._stop:
                break
            time.sleep(self._reconnect_seconds)

    async def _connect_and_pump(self):  # noqa: C901 - acceptable complexity
        async for ws in websockets.connect(self._url):  # auto-reconnect pattern
            self._ws = ws
            print(f"InferenceClient: connected to {self._url}")
            try:
                # Receive loop
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue
                    try:
                        mtype, payload = unwrap_message(msg)
                    except Exception:
                        # Could be a streaming fragment (AppendItemContentText wrapper)
                        t = msg.get("type")
                        if t == "AppendItemContentText":
                            pld = msg.get("payload") or {}
                            frag = {
                                "__stream_fragment__": True,
                                "item_id": pld.get("item_id"),
                                "text": pld.get("text", ""),
                            }
                            self._inference_updates.push(frag)
                        continue
                    if mtype == "Change":
                        try:
                            change = Change.from_safe_delta_dict(payload)
                            self._inference_updates.push(change)
                        except Exception:
                            continue
                    elif mtype == "AppendItemContentText":
                        frag = {
                            "__stream_fragment__": True,
                            "item_id": payload.get("item_id"),
                            "text": payload.get("text", ""),
                        }
                        self._inference_updates.push(frag)
            except websockets.ConnectionClosed:  # pragma: no cover
                print("InferenceClient: websocket closed; reconnecting…")
                continue
            finally:
                try:
                    await ws.close()
                except Exception:
                    pass

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
