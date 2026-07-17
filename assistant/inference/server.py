"""FastAPI websocket server exposing context CRUD operations.

Uses sivkit's WebSocketSyncService (authority role) to synchronize the
context store with connected clients.  Streaming text chunks travel as
custom ``text_append`` delta ops inside the standard sync protocol.
"""

from __future__ import annotations

import asyncio
import copy
import os
import secrets
from contextlib import asynccontextmanager
from typing import Any

# Force verbose backend logs unless explicitly overridden
os.environ.setdefault("LOGLEVEL", "INFO")

import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sivkit.libs.model import load_from_dict
from sivkit.loop import AsyncioMainLoop, set_main_loop
from sivkit.storage.change import Change
from sivkit.storage.delta import Delta
from sivkit.storage.websocket_sync_service import WebSocketSyncService

from assistant.inference.authority import authority_guard
from assistant.inference.backend_protocol import InferenceBackend
from assistant.inference.context import ContextManager
from assistant.inference.context_store import ContextStore
from assistant.inference.llama_model_proxy import LlamaModelProxy
from assistant.inference.model_manager import ModelManager
from assistant.inference.service import InferenceService, generate_oneshot
from assistant.logging_config import configure_logging
from assistant.model_configs import MODEL_SPECS
from assistant.transcription_service import TRANSCRIPTION_MODELS, TranscriptionService

log = logging.getLogger(__name__)


def _backend_for_model(model_key: str) -> InferenceBackend:
    """Create the appropriate backend instance for a given model key."""
    spec = MODEL_SPECS[model_key]
    if spec.get("backend") == "llama_cpp":
        return LlamaModelProxy()
    return ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    set_main_loop(AsyncioMainLoop())
    app.state.backend = ModelManager()
    app.state.transcription_service = TranscriptionService()
    app.state.active_service = None
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


@app.get("/context/{focus_mode}")
async def get_raw_context(focus_mode: str) -> dict[str, Any]:
    """Return the structured context for a focus mode.

    The response includes full messages with image metadata.
    The client is responsible for formatting display text.
    """
    service: InferenceService | None = app.state.active_service
    if service is None:
        return {"status": "error", "error_message": "No active session"}

    backend: InferenceBackend = app.state.backend
    if backend.state != "loaded":
        return {"status": "error", "error_message": "Model not loaded"}

    ctx = service.context
    batch = ctx.items_as_qwen_chat_messages(focus_mode=focus_mode)
    if not batch.messages:
        return {
            "status": "ok",
            "prompt": "",
            "messages": [],
            "message_count": 0,
            "image_count": 0,
        }

    # Enrich image placeholders with metadata (dimensions from resolved PIL images)
    enriched_messages = _enrich_image_metadata(batch.messages, batch.images)

    # Apply chat template
    prompt_text: str | None = None
    try:
        if isinstance(backend, ModelManager) and backend._processor is not None:
            compiled = compile_qwen_context(
                None, backend._processor, messages_override=batch
            )
            image_token_counts = _get_image_token_counts(
                compiled.processor_inputs, backend._processor
            )
            prompt_text = _replace_image_pads_with_metadata(
                compiled.prompt, batch.images, image_token_counts
            )
    except Exception as exc:
        log.warning("Could not apply chat template: %s", exc)

    return {
        "status": "ok",
        "prompt": prompt_text or "",
        "messages": enriched_messages,
        "message_count": len(batch.messages),
        "image_count": len(batch.images),
    }


def _get_image_token_counts(
    processor_inputs: dict[str, Any], processor: Any
) -> list[int]:
    """Count image_pad tokens per image from the tokenized input_ids."""
    input_ids = processor_inputs.get("input_ids")
    if input_ids is None:
        return []

    tokenizer = processor.tokenizer
    pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")

    ids = input_ids[0].tolist()  # first (only) batch item
    counts: list[int] = []
    i = 0
    while i < len(ids):
        if ids[i] == vision_start_id:
            count = 0
            i += 1
            while i < len(ids) and ids[i] != vision_end_id:
                if ids[i] == pad_id:
                    count += 1
                i += 1
            counts.append(count)
        i += 1
    return counts


def _replace_image_pads_with_metadata(
    prompt: str, images: list[Any], token_counts: list[int]
) -> str:
    """Replace <|vision_start|><|image_pad|><|vision_end|> sequences with image metadata."""
    pattern = r"<\|vision_start\|>((<\|image_pad\|>)+)<\|vision_end\|>"
    img_idx = [0]

    def _replacer(match: re.Match) -> str:
        idx = img_idx[0]
        img_idx[0] += 1
        img = images[idx] if idx < len(images) else None
        tok_count = token_counts[idx] if idx < len(token_counts) else "?"
        if img is not None:
            w, h = img.size
            return f"<|vision_start|>[image {w}x{h}, {tok_count} tokens]<|vision_end|>"
        return match.group(0)

    return re.sub(pattern, _replacer, prompt)


def _enrich_image_metadata(messages: list[dict], images: list[Any]) -> list[dict]:
    """Add width/height to image content entries from resolved PIL images."""
    result = copy.deepcopy(messages)
    img_idx = 0
    for msg in result:
        for part in msg.get("content", []):
            if part.get("type") == "image" and img_idx < len(images):
                w, h = images[img_idx].size
                part["width"] = w
                part["height"] = h
                img_idx += 1
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
    app.state.active_service = service
    store: ContextStore = service.context._store
    pending_tasks: set[asyncio.Task] = set()

    # --- Wire inference trigger on remote changes ---
    sync = WebSocketSyncService(store, role="authority")

    def _on_remote_changes(delta: Delta, origin: str | None = None) -> None:
        if origin != "remote":
            return
        for change in delta.changes():
            # Check for cancellation signals on any change (even updates to dispatched items)
            service.check_cancellation(change)

            # Supervision: a pass/correct verdict on a held turn resumes its
            # withheld continuation.
            resume_item = service.resume_target(change)
            if resume_item is not None:
                rtask = asyncio.create_task(service._resume_held_turn(resume_item))
                pending_tasks.add(rtask)
                rtask.add_done_callback(pending_tasks.discard)
                rtask.add_done_callback(
                    lambda t: (
                        log.error(
                            "WS session=%s resume task exception: %s",
                            session_id,
                            t.exception(),
                        )
                        if not t.cancelled() and t.exception()
                        else None
                    )
                )

            if change.is_delete():
                continue
            if not change.forward_component:
                continue

            async def _safe_handle(c: Change = change) -> None:
                try:
                    await service.handle_change(c)
                except asyncio.CancelledError:
                    log.info("WS session=%s task cancelled", session_id)
                except Exception as exc:  # noqa: BLE001
                    log.error(
                        "WS session=%s inference handle_change failed: %s",
                        session_id,
                        exc,
                        exc_info=True,
                    )

            task = asyncio.create_task(_safe_handle())
            pending_tasks.add(task)
            task.add_done_callback(pending_tasks.discard)
            task.add_done_callback(
                lambda t: (
                    log.error(
                        "WS session=%s task exception: %s",
                        session_id,
                        t.exception(),
                    )
                    if not t.cancelled() and t.exception()
                    else None
                )
            )

    store.add_on_changes_callback(_on_remote_changes)

    # Register the write-authority guard FIRST so it fires before any
    # propagation. It raises ValueError if this process (the inference server)
    # writes a field it does not own.
    store.add_on_changes_callback(
        lambda d, o: authority_guard("inference-server", d, o)
    )

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

        # Signal all active generations to stop
        for event in service._cancel_events.values():
            event.set()

        # Cancel and await all pending inference tasks
        for task in pending_tasks:
            task.cancel()
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)

        if app.state.active_service is service:
            app.state.active_service = None
        log.info("WS session=%s ended", session_id)
