"""FastAPI websocket server exposing context CRUD operations.

Overhauled to:
- Use a module-level FastAPI `app` with lifespan setup
- Exchange only fusion Change objects over the websocket (no wrappers)
- Notify an InferenceService on client-originated updates
"""

from __future__ import annotations

import asyncio
import secrets
from contextlib import asynccontextmanager
from typing import Any, Optional

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fusion import get_logger
from fusion.libs.entity.change import Change
from fusion.loop import AsyncioMainLoop, set_main_loop
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from assistant.inference.interface import (
    AppendItemContentTextMessage,
    ChangeMessage,
    parse_message,
    wrap_change,
)
from assistant.inference.service import InferenceService, ModelConfig

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize shared services
    # Ensure fusion uses asyncio loop
    set_main_loop(AsyncioMainLoop())

    # Configure model from env if needed (kept simple for now)
    model_config = ModelConfig()
    processor = AutoProcessor.from_pretrained(model_config.model_id)
    # Eagerly load model with appropriate dtype/device
    device = torch.device(model_config.device)

    def _select_dtype(device: torch.device, precision: Optional[str]) -> torch.dtype:
        if precision == "bf16":
            return torch.bfloat16
        if precision == "fp16":
            return torch.float16
        if precision == "fp32":
            return torch.float32
        if device.type == "cuda":
            return torch.float16
        return torch.float32

    dtype = _select_dtype(device, model_config.precision)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_config.model_id, torch_dtype=dtype
    )
    _to = getattr(model, "to")
    _to(device)
    app.state.model_config = model_config
    app.state.processor = processor
    app.state.model = model
    try:
        yield
    finally:
        pass


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "service": "inference"}


@app.websocket("/ws/context")
async def context_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    model_config: ModelConfig = app.state.model_config
    processor: AutoProcessor = app.state.processor
    model: Qwen2_5_VLForConditionalGeneration = app.state.model
    session_id = secrets.token_hex(4)
    service = InferenceService(
        processor,
        model,
        model_config,
        session_id=session_id,
    )

    def _on_client_update(change: Change) -> None:
        asyncio.create_task(service.handle_change(change))

    client_sub = service.client_updates.subscribe(_on_client_update)

    async def _safe_send_json(payload: Any) -> None:
        try:
            await websocket.send_json(payload)
        except Exception as exc:  # noqa: BLE001
            log.error(f"Failed to send WS message: {exc}", exc_info=True)

    # Send current state as a series of CREATE changes
    for item in service.context.list_items():
        await _safe_send_json(wrap_change(Change.CREATE(item)))

    # Subscribe to inference updates and forward to websocket
    def _on_inference_update(evt):  # evt: Change | AppendItemContentTextMessage
        async def _forward() -> None:
            if isinstance(evt, Change):
                await _safe_send_json(wrap_change(evt))
            elif isinstance(evt, dict) and evt.get("type") == "AppendItemContentText":
                await _safe_send_json(evt)
            else:
                log.warning(f"Unknown inference update event type: {evt!r}")

        asyncio.create_task(_forward())

    inference_sub = service.inference_updates.subscribe(_on_inference_update)

    try:
        while True:
            try:
                raw = await websocket.receive_json()
            except WebSocketDisconnect as exc:  # normal close
                code = getattr(exc, "code", None)
                reason = getattr(exc, "reason", "")
                if code == 1000:
                    log.info(f"WebSocket closed normally code={code} reason={reason}")
                else:
                    log.warning(f"WebSocket disconnected code={code} reason={reason}")
                break
            except Exception as exc:  # noqa: BLE001
                log.error(f"Error receiving WS message: {exc}", exc_info=True)
                break
            # Accept both wrapped protocol and legacy raw Change safe-delta
            try:
                inbound = parse_message(raw)
                if inbound["type"] != "Change":
                    raise Exception("Unsupported message type received")
                change = Change.from_safe_delta_dict(
                    inbound["payload"]
                )  # payload guaranteed dict
            except Exception as exc:  # noqa: BLE001
                log.error(f"Invalid message wrapper: {exc}", exc_info=True)
                await _safe_send_json({"error": str(exc)})
                continue

            try:
                repo_change = await service.context.apply_change(change)
                service.client_updates.push(repo_change)
            except Exception as exc:  # noqa: BLE001
                log.error(f"Failed to apply change: {exc}", exc_info=True)
                await _safe_send_json({"error": f"apply_change: {exc}"})
                continue
    finally:
        inference_sub.unsubscribe()
        client_sub.unsubscribe()
