from __future__ import annotations

import asyncio
import gc
import logging
import threading
from collections.abc import AsyncIterator
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, TextIteratorStreamer

from assistant.inference.qwen_tokens import compile_qwen_context
from assistant.model_configs import MODEL_SPECS

log = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, device: str | None = None):
        self._model: Any | None = None
        self._processor: AutoProcessor | None = None
        self._model_key: str | None = None
        self._state: str = "unloaded"  # "unloaded" | "loading" | "loaded"
        self._device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._load_lock = asyncio.Lock()
        self._model_ready = asyncio.Event()
        self._model_ready.set()

    @property
    def state(self) -> str:
        return self._state

    @property
    def current_model_key(self) -> str | None:
        return self._model_key

    @property
    def device(self) -> torch.device:
        return self._device

    def _select_dtype(self, precision: str | None = None) -> torch.dtype:
        if precision == "bf16":
            return torch.bfloat16
        if precision == "fp16":
            return torch.float16
        if precision == "fp32":
            return torch.float32
        if self._device.type == "cuda":
            return torch.float16
        return torch.float32

    async def load_model(self, model_key: str, *, precision: str | None = None) -> None:
        if model_key not in MODEL_SPECS:
            raise ValueError(f"Unknown model key: {model_key}")

        async with self._load_lock:
            if self._model_key == model_key and self._state == "loaded":
                log.info("Model %s already loaded", model_key)
                return

            self._model_ready.clear()
            self._state = "loading"
            log.info("Loading model %s ...", model_key)

            try:
                self._release_model()

                spec = MODEL_SPECS[model_key]
                model_id = spec["id"]
                model_class = spec["class"]
                dtype = self._select_dtype(precision)

                processor = await asyncio.to_thread(
                    AutoProcessor.from_pretrained, model_id
                )
                model = await asyncio.to_thread(
                    model_class.from_pretrained, model_id, dtype=dtype
                )
                _to = getattr(model, "to")
                await asyncio.to_thread(_to, self._device)

                self._processor = processor
                self._model = model
                self._model_key = model_key
                self._state = "loaded"
                log.info("Model %s loaded successfully", model_key)
            except Exception:
                self._state = "unloaded"
                self._model_key = None
                self._model = None
                self._processor = None
                log.error("Failed to load model %s", model_key, exc_info=True)
                raise
            finally:
                self._model_ready.set()

    async def unload_model(self) -> None:
        async with self._load_lock:
            self._model_ready.clear()
            self._release_model()
            self._state = "unloaded"
            self._model_key = None
            log.info("Model unloaded")
            self._model_ready.set()

    def _release_model(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    async def get_model(self) -> tuple[Any, AutoProcessor]:
        await self._model_ready.wait()
        if self._model is None or self._processor is None:
            raise RuntimeError("No model loaded")
        return self._model, self._processor

    def get_state_dict(self) -> dict:
        return {
            "model_key": self._model_key,
            "state": self._state,
        }

    # --- InferenceBackend generate methods ---

    ALLOWED_MODEL_INPUT_KEYS = {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "image_position_ids",
        "mm_token_type_ids",
        "token_type_ids",
    }

    async def generate(
        self,
        messages: list[dict[str, Any]],
        images: list[Image.Image],
        gen_params: dict[str, Any],
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Non-streaming generation. Returns {status, text, tokens}."""
        model, processor = await self.get_model()

        from assistant.inference.context import MessageBatch

        # Build a temporary ContextManager and compile
        compiled = compile_qwen_context(
            None,
            processor,
            chat_template_kwargs=chat_template_kwargs,
            messages_override=MessageBatch(messages=messages, images=images),
        )
        if not compiled.messages:
            return {"status": "error", "error_message": "No messages in context"}

        inputs = {
            k: v.to(self._device) if isinstance(v, torch.Tensor) else v
            for k, v in compiled.processor_inputs.items()
        }
        model_kwargs = {
            k: v
            for k, v in inputs.items()
            if k in self.ALLOWED_MODEL_INPUT_KEYS and v is not None
        }

        params = dict(gen_params)
        temperature = params.get("temperature")
        if temperature is not None:
            if float(temperature) > 0.0:
                params["do_sample"] = True
            else:
                params.pop("temperature", None)

        try:
            model.eval()
            with torch.no_grad():
                generation = model.generate(**model_kwargs, **params)
            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = generation[:, prompt_len:]
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is None:
                raise RuntimeError("Processor missing tokenizer for decoding")
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[
                0
            ].strip()
            tokens_len = (
                int(new_tokens.shape[1]) if hasattr(new_tokens, "shape") else None
            )
            log.info(
                "generate complete text_len=%d tokens=%s", len(decoded), tokens_len
            )
            return {"status": "success", "text": decoded, "tokens": tokens_len}
        except Exception as exc:
            log.error("generate failed: %s", exc, exc_info=True)
            return {"status": "error", "error_message": str(exc)}

    async def generate_stream(
        self,
        messages: list[dict[str, Any]],
        images: list[Image.Image],
        gen_params: dict[str, Any],
        chat_template_kwargs: dict[str, Any] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> AsyncIterator[str]:
        """Streaming generation. Yields text chunks."""
        model, processor = await self.get_model()

        from assistant.inference.context import MessageBatch

        compiled = compile_qwen_context(
            None,
            processor,
            chat_template_kwargs=chat_template_kwargs,
            messages_override=MessageBatch(messages=messages, images=images),
        )
        if not compiled.messages:
            return

        inputs = {
            k: v.to(self._device) if isinstance(v, torch.Tensor) else v
            for k, v in compiled.processor_inputs.items()
        }
        model_kwargs = {
            k: v
            for k, v in inputs.items()
            if k in self.ALLOWED_MODEL_INPUT_KEYS and v is not None
        }

        params = dict(gen_params)
        temperature = params.get("temperature")
        if temperature is not None:
            if float(temperature) > 0.0:
                params["do_sample"] = True
            else:
                params.pop("temperature", None)

        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Processor has no tokenizer for streaming")

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        model.eval()

        loop = asyncio.get_running_loop()
        chunk_queue: asyncio.Queue[str | object] = asyncio.Queue()
        sentinel = object()

        def _run_generate() -> None:
            try:
                with torch.no_grad():
                    model.generate(**model_kwargs, **params, streamer=streamer)
            except Exception as e:
                log.error("Streaming generation thread error: %s", e, exc_info=True)

        def _drain_streamer() -> None:
            try:
                for chunk in streamer:
                    loop.call_soon_threadsafe(chunk_queue.put_nowait, chunk)
            except Exception as exc:
                log.error("Streamer drain error: %s", exc, exc_info=True)
            finally:
                loop.call_soon_threadsafe(chunk_queue.put_nowait, sentinel)

        thread = threading.Thread(target=_run_generate, daemon=True)
        thread.start()
        drain_thread = threading.Thread(target=_drain_streamer, daemon=True)
        drain_thread.start()

        try:
            while True:
                chunk = await chunk_queue.get()
                if chunk is sentinel:
                    break
                if cancel_event and cancel_event.is_set():
                    break
                if not chunk:
                    continue
                yield str(chunk)
        finally:
            if thread.is_alive():
                thread.join(timeout=0.1)
            if drain_thread.is_alive():
                drain_thread.join(timeout=0.1)
