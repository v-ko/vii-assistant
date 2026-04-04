"""FastAPI websocket server exposing context CRUD operations.

Overhauled to:
- Use a module-level FastAPI `app` with lifespan setup
- Exchange only fusion Change objects over the websocket (no wrappers)
- Notify an InferenceService on client-originated updates
- Dynamic model load/unload via ModelManager
"""

from __future__ import annotations

import asyncio
import os
import secrets
from contextlib import asynccontextmanager
from typing import Any

# Force verbose backend logs unless explicitly overridden
os.environ.setdefault("LOGLEVEL", "INFO")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fusion import get_logger
from fusion.libs.entity.change import Change
from fusion.loop import AsyncioMainLoop, set_main_loop
from pydantic import BaseModel

from assistant.inference.context import ContextItem
from assistant.inference.interface import parse_message, wrap_change
from assistant.inference.model_manager import ModelManager
from assistant.inference.service import InferenceService
from assistant.model_configs import MODEL_SPECS

log = get_logger(__name__)


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    set_main_loop(AsyncioMainLoop())
    app.state.model_manager = ModelManager()
    try:
        yield
    finally:
        await app.state.model_manager.unload_model()


app = FastAPI(lifespan=lifespan)


class LoadModelRequest(BaseModel):
    model_key: str | None = None


@app.get("/status")
async def status() -> dict[str, Any]:
    mm: ModelManager = app.state.model_manager
    return {
        "status": "ok",
        "service": "inference",
        "model": mm.get_state_dict(),
    }


@app.post("/model/load")
async def load_model(body: LoadModelRequest) -> dict[str, Any]:
    mm: ModelManager = app.state.model_manager
    model_key = body.model_key

    if model_key is None or model_key == "none":
        await mm.unload_model()
        return {"status": "ok", "model": mm.get_state_dict()}

    if model_key not in MODEL_SPECS:
        return {"status": "error", "message": f"Unknown model key: {model_key}"}

    # Fire-and-forget: start loading in background so the HTTP response is immediate
    asyncio.create_task(mm.load_model(model_key))
    return {"status": "accepted", "model": mm.get_state_dict()}


@app.websocket("/ws/context")
async def context_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    model_manager: ModelManager = app.state.model_manager
    session_id = secrets.token_hex(4)
    log.info("WS accepted session=%s", session_id)

    service = InferenceService(
        model_manager,
        session_id=session_id,
    )

    # TODO: review this method
    def _on_client_update(change: Change) -> None:
        log.info(
            "WS session=%s client_update forwarded to inference %s",
            session_id,
            _summarize_change(change),
        )

        async def _safe_handle() -> None:
            try:
                await service.handle_change(change)
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "WS session=%s inference handle_change failed: %s",
                    session_id,
                    exc,
                    exc_info=True,
                )

        task = asyncio.create_task(_safe_handle())
        task.add_done_callback(
            lambda t: (
                log.error(
                    "WS session=%s inference task exception: %s",
                    session_id,
                    t.exception(),
                )
                if t.exception()
                else None
            )
        )

    client_sub = service.client_updates.subscribe(_on_client_update)

    async def _safe_send_json(payload: Any) -> None:
        try:
            await websocket.send_json(payload)
        except Exception as exc:  # noqa: BLE001
            log.error(f"Failed to send WS message: {exc}", exc_info=True)

    # Send current state as a series of CREATE changes
    initial_count = 0
    for item in service.context.items_sorted():
        initial_count += 1
        await _safe_send_json(wrap_change(Change.CREATE(item)))
    log.info("WS session=%s sent %d initial context items", session_id, initial_count)

    # Subscribe to inference updates and forward to websocket
    def _on_inference_update(evt):  # evt: Change | AppendItemContentTextMessage
        async def _forward() -> None:
            if isinstance(evt, Change):
                log.info(
                    "WS session=%s outbound inference change %s",
                    session_id,
                    _summarize_change(evt),
                )
                await _safe_send_json(wrap_change(evt))
            elif isinstance(evt, dict) and evt.get("type") == "AppendItemContentText":
                log.info(
                    "WS session=%s outbound append fragment item_id=%s text_len=%d",
                    session_id,
                    evt.get("payload", {}).get("item_id"),
                    len(evt.get("payload", {}).get("text", "")),
                )
                await _safe_send_json(evt)
            else:
                log.warning(f"Unknown inference update event type: {evt!r}")

        asyncio.create_task(_forward())

    inference_sub = service.inference_updates.subscribe(_on_inference_update)

    try:
        while True:
            try:
                raw = await websocket.receive_json()
                # Avoid logging large blobs; log shape/keys only
                rtype = raw.get("type") if isinstance(raw, dict) else type(raw)
                payload = raw.get("payload") if isinstance(raw, dict) else None
                payload_keys = list(payload.keys()) if isinstance(payload, dict) else []
                log.info(
                    "WS session=%s recv type=%s payload_keys=%s",
                    session_id,
                    rtype,
                    payload_keys,
                )
            except WebSocketDisconnect as exc:  # normal close
                code = getattr(exc, "code", None)
                reason = getattr(exc, "reason", "")
                if code == 1000:
                    log.info(f"WebSocket closed normally code={code} reason={reason}")
                else:
                    log.warning(f"WebSocket disconnected code={code} reason={reason}")
                break
            except Exception as exc:  # noqa: BLE001
                # Log the raw text that failed to parse
                try:
                    raw_text = await websocket.receive_text()
                    log.error(
                        f"Error receiving WS message: {exc}\nRaw message: {raw_text!r}",
                        exc_info=True,
                    )
                except Exception:  # noqa: BLE001
                    log.error(f"Error receiving WS message: {exc}", exc_info=True)
                # Send error to client and continue (don't break the connection)
                await _safe_send_json({"error": f"receive_json: {exc}"})
                continue
            # Accept both wrapped protocol and legacy raw Change safe-delta
            try:
                inbound = parse_message(raw)
                if inbound["type"] != "Change":
                    raise Exception("Unsupported message type received")
                change = Change.from_safe_delta_dict(
                    inbound["payload"]
                )  # payload guaranteed dict
            except Exception as exc:  # noqa: BLE001
                log.error(
                    f"Invalid message wrapper: {exc}\nRaw: {raw!r}", exc_info=True
                )
                await _safe_send_json({"error": str(exc)})
                continue

            try:
                log.info(
                    "WS session=%s inbound change %s",
                    session_id,
                    _summarize_change(change),
                )
                repo_change = await service.context.apply_change(change)
                log.info(
                    "WS session=%s applied change %s",
                    session_id,
                    _summarize_change(repo_change),
                )
                service.client_updates.push(repo_change)
            except Exception as exc:  # noqa: BLE001
                log.error(
                    f"Failed to apply change: {exc}\nChange: {change!r}", exc_info=True
                )
                # Create an error context item to show the error to the user
                error_item = ContextItem()
                error_item.position = service.context.next_position()
                error_item.size = 0
                error_item.content = {
                    "text": (
                        f"Error: {exc}\n\nFailed to apply change:"
                        f" {change.change_type.name}"
                    )
                }
                error_item.metadata = {"origin": "system_error"}
                error_change = service.context.insert(error_item)
                await _safe_send_json(wrap_change(error_change))
                # Also send error response
                await _safe_send_json({"error": f"apply_change: {exc}"})
                continue
    finally:
        try:
            inference_sub.unsubscribe()
            client_sub.unsubscribe()
        finally:
            log.info("WS session=%s closed", session_id)
