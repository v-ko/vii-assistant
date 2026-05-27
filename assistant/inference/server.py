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
from fusion.storage.websocket_sync_service import WebSocketSyncService
from pydantic import BaseModel

from assistant.inference.backend_protocol import InferenceBackend
from assistant.inference.context import ContextItem, ContextManager
from assistant.inference.context_store import ContextStore
from assistant.inference.llama_model_proxy import LlamaModelProxy
from assistant.inference.model_manager import ModelManager
from assistant.inference.service import InferenceService, generate_oneshot
from assistant.model_configs import MODEL_SPECS
from assistant.transcription_service import TRANSCRIPTION_MODELS, TranscriptionService

log = get_logger(__name__)


def _backend_for_model(model_key: str) -> InferenceBackend:
    """Create the appropriate backend instance for a given model key."""
    spec = MODEL_SPECS[model_key]
    if spec.get("backend") == "llama_cpp":
        return LlamaModelProxy()
    return ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    set_main_loop(AsyncioMainLoop())
    app.state.backend: InferenceBackend = ModelManager()
    app.state.transcription_service = TranscriptionService()
    try:
        yield
    finally:
        await app.state.backend.unload_model()


app = FastAPI(lifespan=lifespan)


class LoadModelRequest(BaseModel):
    model_key: str | None = None


@app.get("/status")
async def status() -> dict[str, Any]:
    backend: InferenceBackend = app.state.backend
    return {
        "status": "ok",
        "service": "inference",
        "model": backend.get_state_dict(),
    }


@app.post("/model")
async def load_model(body: LoadModelRequest) -> dict[str, Any]:
    backend: InferenceBackend = app.state.backend
    model_key = body.model_key

    if not model_key:
        return {"status": "error", "message": "model_key is required"}

    if model_key not in MODEL_SPECS:
        return {"status": "error", "message": f"Unknown model key: {model_key}"}

    # Switch backend type if needed
    spec = MODEL_SPECS[model_key]
    needs_llama = spec.get("backend") == "llama_cpp"
    is_llama = isinstance(backend, LlamaModelProxy)

    if needs_llama and not is_llama:
        await backend.unload_model()
        backend = LlamaModelProxy()
        app.state.backend = backend
    elif not needs_llama and is_llama:
        await backend.unload_model()
        backend = ModelManager()
        app.state.backend = backend

    asyncio.create_task(backend.load_model(model_key))
    return {"status": "accepted", "model": backend.get_state_dict()}


@app.delete("/model")
async def unload_model() -> dict[str, Any]:
    backend: InferenceBackend = app.state.backend
    await backend.unload_model()
    return {"status": "ok", "model": backend.get_state_dict()}


class InferRequest(BaseModel):
    context_data: list[dict[str, Any]]
    generation_params: dict[str, Any] | None = None
    chat_template_params: dict[str, Any] | None = None


@app.post("/infer")
async def infer(body: InferRequest) -> dict[str, Any]:
    """Stateless single-shot inference from serialized context items."""
    backend: InferenceBackend = app.state.backend
    if backend.state != "loaded":
        return {"status": "error", "error_message": "No model loaded"}

    # Hydrate a throwaway ContextManager from the provided entity dicts
    ctx = ContextManager()
    entities = [load_from_dict(d) for d in body.context_data]
    ctx.store.load_data(entities)

    result = await generate_oneshot(
        ctx,
        backend,
        gen_params=body.generation_params,
        chat_template_params=body.chat_template_params,
    )
    return result


class TranscribeRequest(BaseModel):
    model_type: str = "parakeet-tdt-0.6b-v3-int8"
    model_dir: str | None = None
    sample_rate: int = 16000
    audio_b64: str  # base64-encoded float32 PCM audio


@app.post("/transcribe")
async def transcribe_audio(body: TranscribeRequest) -> dict[str, Any]:
    """Run chunk-level ASR on base64 float32 PCM audio.

    The audio should be mono, float32 little-endian PCM samples encoded in base64.
    Chunking and stitching are client-side responsibilities.
    """
    import base64

    import numpy as np

    svc: TranscriptionService = app.state.transcription_service

    if body.model_type not in TRANSCRIPTION_MODELS:
        return {
            "status": "error",
            "error_message": f"Unknown model type: {body.model_type}. "
            f"Available: {list(TRANSCRIPTION_MODELS.keys())}",
        }

    try:
        raw = base64.b64decode(body.audio_b64)
    except Exception:
        return {"status": "error", "error_message": "Invalid base64 audio data"}

    audio = np.frombuffer(raw, dtype=np.float32)

    try:
        result = await asyncio.to_thread(
            svc.transcribe_chunk,
            audio,
            body.sample_rate,
            body.model_type,
            body.model_dir,
        )
    except Exception as exc:
        log.error("Transcription failed: %s", exc, exc_info=True)
        return {"status": "error", "error_message": str(exc)}

    return {
        "status": "success",
        "text": result.text,
        "words": [
            {"word": w.word, "start": w.start, "end": w.end} for w in result.words
        ],
    }


@app.websocket("/ws/context")
async def context_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    backend: InferenceBackend = app.state.backend
    session_id = secrets.token_hex(4)
    log.info("WS accepted session=%s", session_id)

    service = InferenceService(backend, session_id=session_id)
    store: ContextStore = service.context._store

    # --- Wire inference trigger on remote changes ---
    sync = WebSocketSyncService(store, role="authority")

    def _on_remote_changes(delta: Delta, origin: str | None = None) -> None:
        if origin != "remote":
            return
        for change in delta.changes():
            # Check for cancellation signals on any change (even updates to dispatched items)
            service.check_cancellation(change)

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

    store.add_on_changes_callback(_on_remote_changes)

    # --- Run the sync protocol ---
    async def send(msg: dict) -> None:
        await websocket.send_json(msg)

    async def receive() -> dict:
        try:
            return await websocket.receive_json()
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
                    "WS session=%s disconnected code=%s reason=%s",
                    session_id,
                    code,
                    reason,
                )
            raise asyncio.CancelledError from exc

    try:
        await sync.run(send, receive)
    except asyncio.CancelledError:
        pass
    except Exception as exc:  # noqa: BLE001
        log.error("WS session=%s error: %s", session_id, exc, exc_info=True)
    finally:
        store.remove_on_changes_callback(_on_remote_changes)
        log.info("WS session=%s ended", session_id)
