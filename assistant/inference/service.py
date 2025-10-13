"""Inference service that reacts to context changes and produces model outputs."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, cast

import torch
from fusion import get_logger
from fusion.libs.channel import Channel
from fusion.libs.entity.change import Change
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TextIteratorStreamer,
)

from .context import ContextItem, ContextManager
from .qwen_tokens import compile_qwen_context

logger = get_logger(__name__)


@dataclass
class ModelConfig:
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: Optional[str] = None  # one of: bf16, fp16, fp32, or None for auto
    default_generation_params: Dict[str, Any] | None = None


class InferenceService:
    def __init__(
        self,
        context: ContextManager,
        processor: AutoProcessor,
        model: Qwen2_5_VLForConditionalGeneration,
        config: Optional[ModelConfig] = None,
    ) -> None:
        self.context = context
        self.config = config or ModelConfig()
        self.processor: AutoProcessor = processor
        self.model: Qwen2_5_VLForConditionalGeneration = model
        self._device = torch.device(self.config.device)
        self._lock = asyncio.Lock()
        # Active streaming threads (keyed by item id) for future cancellation support
        self._active_streams: dict[str, threading.Thread] = {}
        # Channel to publish both full changes and streaming fragments
        self.inference_updates: Optional[Channel] = None

    # No startup needed: model and processor are provided via constructor

    async def handle_change(self, change: Change) -> None:
        # Only react to new/updated items coming from client channel
        if change.is_delete():
            return
        if not change.new_state:
            return
        if not hasattr(change.new_state, "request"):
            logger.info("per-item tokenization not implemented yet")
            return
        item = change.new_state
        if not isinstance(item, ContextItem):
            raise TypeError(f"Expected ContextItem, got {type(item).__name__}")
        if not item.request:
            logger.info(
                "Ignoring context item %s without request", getattr(item, "id", None)
            )
            return
        if isinstance(item.request, dict) and "result" in item.request:
            logger.warning(
                "Request already has a result; ignoring item %s",
                getattr(item, "id", None),
            )
            return

        async with self._lock:
            await self._dispatch_generation(item)

    @staticmethod
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

    # --- Internal helpers ---

    async def _dispatch_generation(self, item: ContextItem) -> None:
        processor = self.processor
        compiled = compile_qwen_context(self.context, processor)

        # Prepare inputs
        inputs = {k: v for k, v in compiled.processor_inputs.items()}
        inputs = {
            k: v.to(self._device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        allowed = {
            "input_ids",
            "attention_mask",
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
        }
        model_kwargs: Dict[str, Any] = {
            k: v for k, v in inputs.items() if k in allowed and v is not None
        }

        gen_params: Dict[str, Any] = {
            "max_new_tokens": 128,
            "do_sample": False,
        }
        if self.config.default_generation_params:
            gen_params.update(self.config.default_generation_params)
        if item.request:
            gen_params.update(item.request)
        # Remove non-generation control keys so HF generate doesn't error
        for _k in ("stream", "completed", "result", "error_message"):
            gen_params.pop(_k, None)
        temperature = gen_params.get("temperature")
        if temperature is not None:
            temp_f = float(temperature)
            if temp_f > 0.0:
                gen_params["do_sample"] = True
            else:
                gen_params.pop("temperature", None)

        updated = cast(ContextItem, item.copy())
        stream = bool((item.request or {}).get("stream"))
        if stream:
            await self._generate_stream(updated, model_kwargs, gen_params)
        else:
            await self._generate_non_stream(updated, inputs, model_kwargs, gen_params)

    async def _generate_non_stream(
        self,
        updated: ContextItem,
        inputs: Dict[str, Any],
        model_kwargs: Dict[str, Any],
        gen_params: Dict[str, Any],
    ) -> None:
        try:
            model = self.model
            model.eval()
            with torch.no_grad():
                generation = model.generate(**model_kwargs, **gen_params)
            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = generation[:, prompt_len:]
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer is None:
                raise RuntimeError("Processor missing tokenizer for decoding")
            decoded_full = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[
                0
            ].strip()
            content = dict(updated.content or {})
            content["text"] = decoded_full
            updated.content = content
            req = dict(updated.request or {})
            req["result"] = "success"
            req["completed"] = True
            updated.request = req
            # tokens length
            try:
                tokens_len = (
                    int(new_tokens.shape[1]) if hasattr(new_tokens, "shape") else None
                )
            except Exception:
                tokens_len = None
            if tokens_len is not None:
                meta = dict(updated.metadata or {})
                meta["tokens_len"] = tokens_len
                updated.metadata = meta
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Inference failed (non-stream) for item %s: %s",
                getattr(updated, "id", None),
                exc,
                exc_info=True,
            )
            req = dict(updated.request or {})
            req["result"] = "error"
            req["error_message"] = str(exc)
            req["completed"] = True
            updated.request = req
        finally:
            ch = self.context.update(updated)
            if self.inference_updates:
                self.inference_updates.push(ch)

    async def _generate_stream(
        self,
        updated: ContextItem,
        model_kwargs: Dict[str, Any],
        gen_params: Dict[str, Any],
    ) -> None:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Processor has no tokenizer for streaming")
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        model = self.model
        model.eval()

        def _run_generate() -> None:
            try:
                with torch.no_grad():
                    model.generate(**model_kwargs, **gen_params, streamer=streamer)
            except Exception as e:  # noqa: BLE001
                logger.error(
                    "Streaming generation thread error for %s: %s",
                    updated.id,
                    e,
                    exc_info=True,
                )

        thread = threading.Thread(target=_run_generate, daemon=True)
        uid = str(updated.id)
        self._active_streams[uid] = thread
        thread.start()

        pieces: list[str] = []
        loop = asyncio.get_running_loop()
        try:
            while True:
                if not thread.is_alive() and getattr(streamer, "text_queue").qsize() == 0:  # type: ignore[attr-defined]
                    break
                try:
                    chunk = await loop.run_in_executor(None, next, streamer)
                except StopIteration:
                    await asyncio.sleep(0)
                    continue
                if not chunk:
                    await asyncio.sleep(0)
                    continue
                pieces.append(chunk)
                if self.inference_updates:
                    # Streaming fragment event distinguished at websocket layer by wrapper
                    self.inference_updates.push(
                        {
                            "__stream_fragment__": True,
                            "item_id": updated.id,
                            "text": chunk,
                        }
                    )
                await asyncio.sleep(0)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Inference failed (stream) for item %s: %s",
                getattr(updated, "id", None),
                exc,
                exc_info=True,
            )
            req = dict(updated.request or {})
            req["result"] = "error"
            req["error_message"] = str(exc)
            req["completed"] = True
            updated.request = req
        else:
            full_text = "".join(pieces).strip()
            content = dict(updated.content or {})
            content["text"] = full_text
            updated.content = content
            req = dict(updated.request or {})
            req["result"] = "success"
            req["completed"] = True
            updated.request = req
            try:
                encoded = tokenizer(full_text, add_special_tokens=False)  # type: ignore[call-arg]
                tokens_len = (
                    len(encoded["input_ids"])
                    if isinstance(encoded, dict) and "input_ids" in encoded
                    else None
                )
            except Exception:
                tokens_len = None
            if tokens_len is not None:
                meta = dict(updated.metadata or {})
                meta["tokens_len"] = tokens_len
                updated.metadata = meta
        finally:
            self._active_streams.pop(uid, None)
            ch = self.context.update(updated)
            if self.inference_updates:
                self.inference_updates.push(ch)
