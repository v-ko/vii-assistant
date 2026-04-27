"""FastAPI websocket server exposing context CRUD operations.

Uses fusion's WebSocketSyncService (authority role) to synchronize the
context store with connected clients.  Streaming text chunks travel as
custom ``text_append`` delta ops inside the standard sync protocol.
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
from fusion.libs.model import load_from_dict
from fusion.loop import AsyncioMainLoop, set_main_loop
from fusion.storage.change import Change
from fusion.storage.delta import Delta
from fusion.storage.ws_sync_service import WebSocketSyncService
from pydantic import BaseModel

from assistant.inference.context import ContextItem, ContextManager
from assistant.inference.context_store import ContextStore
from assistant.inference.model_manager import ModelManager
from assistant.inference.service import InferenceService
from assistant.model_configs import MODEL_SPECS

log = get_logger(__name__)


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


@app.post("/model")
async def load_model(body: LoadModelRequest) -> dict[str, Any]:
    mm: ModelManager = app.state.model_manager
    model_key = body.model_key

    if not model_key:
        return {"status": "error", "message": "model_key is required"}

    if model_key not in MODEL_SPECS:
        return {"status": "error", "message": f"Unknown model key: {model_key}"}

    asyncio.create_task(mm.load_model(model_key))
    return {"status": "accepted", "model": mm.get_state_dict()}


@app.delete("/model")
async def unload_model() -> dict[str, Any]:
    mm: ModelManager = app.state.model_manager
    await mm.unload_model()
    return {"status": "ok", "model": mm.get_state_dict()}


@app.websocket("/ws/context")
async def context_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    model_manager: ModelManager = app.state.model_manager
    session_id = secrets.token_hex(4)
    log.info("WS accepted session=%s", session_id)

    service = InferenceService(model_manager, session_id=session_id)
    store: ContextStore = service.context._repo

    # --- Wire store.on_changes → sync service + inference trigger ---
    sync = WebSocketSyncService(store, role="authority")

    def _on_store_changes(delta: Delta, origin: str | None = None) -> None:
        # Forward to sync service (skips remote-origin automatically).
        # Use _enqueue_delta directly instead of on_store_changes to avoid
        # blocking the event loop (text_append and update_one fire on_changes
        # synchronously on the same thread that runs the WSSS send/receive
        # loops — blocking would deadlock).
        if origin != "remote" and sync._running:
            sync._enqueue_delta(delta.asdict())

        # Trigger inference on remote (client) changes
        if origin == "remote":
            for change in delta.changes():
                if change.is_delete():
                    continue
                if not change.forward_component:
                    continue

                async def _safe_handle(c: Change = change) -> None:
                    try:
                        await service.handle_change(c)
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
                            "WS session=%s task exception: %s",
                            session_id,
                            t.exception(),
                        )
                        if t.exception()
                        else None
                    )
                )

    store.on_changes = _on_store_changes

    # --- Run the sync protocol ---
    async def send(msg: dict) -> None:
        await websocket.send_json(msg)

    async def receive() -> dict:
        return await websocket.receive_json()

    try:
        await sync.run(send, receive)
    except WebSocketDisconnect as exc:
        code = getattr(exc, "code", None)
        reason = getattr(exc, "reason", "")
        if code == 1000:
            log.info(
                "WS session=%s closed normally code=%s reason=%s",
                session_id,
                code,
                reason,
            )
        else:
            log.warning(
                "WS session=%s disconnected code=%s reason=%s", session_id, code, reason
            )
    except Exception as exc:  # noqa: BLE001
        log.error("WS session=%s error: %s", session_id, exc, exc_info=True)
    finally:
        store.on_changes = None
        log.info("WS session=%s ended", session_id)
