"""Inference service that reacts to context changes and produces model outputs."""

from __future__ import annotations

import asyncio
import secrets
import threading
from typing import TYPE_CHECKING, Any, cast

import torch
from fusion import get_logger
from fusion.libs.model import load_from_dict
from fusion.storage.change import Change
from transformers import TextIteratorStreamer

from assistant.inference.context import ContextItem, ContextManager
from assistant.inference.context_store import ContextStore
from assistant.inference.qwen_tokens import compile_qwen_context
from assistant.model_configs import MODEL_SPECS

if TYPE_CHECKING:
    from assistant.inference.model_manager import ModelManager

logger = get_logger(__name__)


class InferenceService:
    def __init__(
        self,
        model_manager: ModelManager,
        session_id: str | None = None,
    ) -> None:
        self.context = ContextManager()
        self.model_manager = model_manager
        self._lock = asyncio.Lock()
        self._active_streams: dict[str, threading.Thread] = {}
        self._dispatched_requests: set[str] = set()
        self.session_id = session_id or secrets.token_hex(4)

    # No startup needed: model and processor are provided via constructor

    async def handle_change(self, change: Change) -> None:
        # Only react to new/updated items coming from client channel
        if change.is_delete():
            return
        if not change.forward_component:
            return
        # Reconstruct the entity from the forward component
        forward = change.forward_component
        if change.is_create():
            item = load_from_dict(dict(forward))
        else:
            # For updates, we need to get the existing entity and apply the forward diff
            existing = self.context._repo.find_one(id=change.entity_id)
            if existing is None:
                logger.info("Entity %s not found for update", change.entity_id)
                return
            from fusion.libs.model import dump_to_dict

            merged = {**dump_to_dict(existing), **forward}
            item = load_from_dict(merged)
        if not isinstance(item, ContextItem):
            raise TypeError(f"Expected ContextItem, got {type(item).__name__}")
        if not hasattr(item, "request") or not item.request:
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

        item_id = str(getattr(item, "id", ""))
        if item_id in self._dispatched_requests:
            logger.info("Skipping already-dispatched request item %s", item_id)
            return
        self._dispatched_requests.add(item_id)

        async with self._lock:
            await self._dispatch_generation(item)

    # --- Internal helpers ---

    def _build_request_params(
        self, request: dict[str, Any] | None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        spec = MODEL_SPECS.get(self.model_manager.current_model_key or "", {})
        gen_params: dict[str, Any] = {
            "max_new_tokens": 128,
            "do_sample": False,
        }
        gen_params.update(spec.get("generation_params") or {})

        chat_template_params = dict(spec.get("chat_template_params") or {})
        if request:
            gen_params.update(request.get("generation_params") or {})
            chat_template_params.update(request.get("chat_template_params") or {})
        return gen_params, chat_template_params

    async def _dispatch_generation(self, item: ContextItem) -> None:
        model, processor = await self.model_manager.get_model()
        device = self.model_manager.device
        gen_params, chat_template_params = self._build_request_params(item.request)
        compiled = compile_qwen_context(
            self.context,
            processor,
            chat_template_kwargs=chat_template_params,
        )
        if not compiled.messages:
            logger.warning(
                "No messages compiled for item=%s, skipping generation",
                getattr(item, "id", None),
            )
            return
        input_ids = compiled.processor_inputs.get("input_ids")
        prompt_tokens = input_ids.shape[1] if input_ids is not None else 0
        logger.info(
            "Dispatching generation for item=%s stream=%s tokens=%d images=%d template_keys=%s",
            getattr(item, "id", None),
            bool((item.request or {}).get("stream")),
            prompt_tokens,
            len(compiled.images),
            sorted(chat_template_params),
        )

        # Prepare inputs
        inputs = {k: v for k, v in compiled.processor_inputs.items()}
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        allowed = {
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
        model_kwargs: dict[str, Any] = {
            k: v for k, v in inputs.items() if k in allowed and v is not None
        }
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
            await self._generate_stream(
                updated, model, processor, model_kwargs, gen_params
            )
        else:
            await self._generate_non_stream(
                updated, model, processor, inputs, model_kwargs, gen_params
            )

    async def _generate_non_stream(
        self,
        updated: ContextItem,
        model: Any,
        processor: Any,
        inputs: dict[str, Any],
        model_kwargs: dict[str, Any],
        gen_params: dict[str, Any],
    ) -> None:
        try:
            model.eval()
            with torch.no_grad():
                generation = model.generate(**model_kwargs, **gen_params)
            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = generation[:, prompt_len:]
            tokenizer = getattr(processor, "tokenizer", None)
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
            logger.info(
                "Non-stream generation complete item=%s text_len=%d tokens=%s",
                getattr(updated, "id", None),
                len(decoded_full),
                tokens_len,
            )
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
            self.context.update(updated)

    async def _generate_stream(
        self,
        updated: ContextItem,
        model: Any,
        processor: Any,
        model_kwargs: dict[str, Any],
        gen_params: dict[str, Any],
    ) -> None:
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Processor has no tokenizer for streaming")
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        model.eval()

        thread_error: list[Exception | None] = [None]
        drain_error: list[Exception | None] = [None]
        loop = asyncio.get_running_loop()
        chunk_queue: asyncio.Queue[str | object] = asyncio.Queue()
        sentinel = object()

        def _run_generate() -> None:
            try:
                with torch.no_grad():
                    model.generate(**model_kwargs, **gen_params, streamer=streamer)
            except Exception as e:  # noqa: BLE001
                thread_error[0] = e
                logger.error(
                    "Streaming generation thread error for %s: %s",
                    updated.id,
                    e,
                    exc_info=True,
                )

        def _drain_streamer() -> None:
            try:
                for chunk in streamer:
                    loop.call_soon_threadsafe(chunk_queue.put_nowait, chunk)
            except Exception as exc:  # noqa: BLE001
                drain_error[0] = exc
                logger.error(
                    "Streamer drain error for %s: %s",
                    updated.id,
                    exc,
                    exc_info=True,
                )
            finally:
                loop.call_soon_threadsafe(chunk_queue.put_nowait, sentinel)

        thread = threading.Thread(target=_run_generate, daemon=True)
        uid = str(updated.id)
        self._active_streams[uid] = thread
        thread.start()

        drain_thread = threading.Thread(target=_drain_streamer, daemon=True)
        drain_thread.start()

        pieces: list[str] = []
        try:
            while True:
                chunk = await chunk_queue.get()
                if chunk is sentinel:
                    break

                if chunk is None:
                    await asyncio.sleep(0)
                    continue

                chunk_s = cast(str, chunk)
                if not chunk_s:
                    await asyncio.sleep(0)
                    continue
                pieces.append(chunk_s)
                # Push streaming chunk as a custom delta op through the store
                store: ContextStore = self.context._repo
                store.text_append(str(updated.id), chunk_s)
                # Give event loop a chance to process outbound messages / cancellation
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
            if thread_error[0] is not None:
                req = dict(updated.request or {})
                req["result"] = "error"
                req["error_message"] = str(thread_error[0])
                req["completed"] = True
                updated.request = req
            else:
                if drain_error[0] is not None:
                    req = dict(updated.request or {})
                    req["result"] = "error"
                    req["error_message"] = str(drain_error[0])
                    req["completed"] = True
                    updated.request = req
                else:
                    if thread.is_alive():
                        thread.join(timeout=0.0)
                    if drain_thread.is_alive():
                        drain_thread.join(timeout=0.0)
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
                logger.info(
                    "Streaming generation complete item=%s text_len=%d tokens=%s",
                    getattr(updated, "id", None),
                    len(full_text),
                    tokens_len,
                )
        finally:
            if thread.is_alive():
                thread.join(timeout=0.1)
            if drain_thread.is_alive():
                drain_thread.join(timeout=0.1)
            self._active_streams.pop(uid, None)
            self.context.update(updated)
