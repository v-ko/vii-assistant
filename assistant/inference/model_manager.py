from __future__ import annotations

import asyncio
import gc
from typing import Any

import torch
from fusion import get_logger
from transformers import AutoProcessor

from assistant.model_configs import MODEL_SPECS

log = get_logger(__name__)


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
