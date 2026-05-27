"""Inference service that reacts to context changes and produces model outputs."""

from __future__ import annotations

import asyncio
import secrets
import threading
from typing import TYPE_CHECKING, Any, cast

from fusion import get_logger
from fusion.libs.model import load_from_dict
from fusion.storage.change import Change

from assistant.inference.context import ContextItem, ContextManager, TextItem
from assistant.inference.context_store import ContextStore
from assistant.model_configs import MODEL_SPECS

if TYPE_CHECKING:
    from assistant.inference.backend_protocol import InferenceBackend

logger = get_logger(__name__)


async def generate_oneshot(
    context_manager: ContextManager,
    backend: "InferenceBackend",
    gen_params: dict[str, Any] | None = None,
    chat_template_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stateless single-shot generation from a populated ContextManager.

    Returns dict with keys: status, text, tokens (on success)
    or status, error_message (on failure).
    """
    spec = MODEL_SPECS.get(backend.current_model_key or "", {})
    params: dict[str, Any] = {"max_new_tokens": 128, "do_sample": False}
    params.update(spec.get("generation_params") or {})
    if gen_params:
        params.update(gen_params)

    template_params = dict(spec.get("chat_template_params") or {})
    if chat_template_params:
        template_params.update(chat_template_params)

    batch = context_manager.items_as_qwen_chat_messages()
    if not batch.messages:
        return {"status": "error", "error_message": "No messages in context"}

    result = await backend.generate(
        batch.messages, batch.images, params, template_params
    )
    if result.get("status") == "success":
        result["bbox_format"] = spec.get("bbox_format", "xyxy")
    return result


class InferenceService:
    def __init__(
        self,
        backend: "InferenceBackend",
        session_id: str | None = None,
    ) -> None:
        self.context = ContextManager()
        self.backend = backend
        self._lock = asyncio.Lock()
        self._cancel_events: dict[str, threading.Event] = {}
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
            existing = self.context._store.find_one(id=change.entity_id)
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

    def cancel_generation(self, item_id: str) -> bool:
        """Signal a running generation to stop. Returns True if it was active."""
        event = self._cancel_events.get(item_id)
        if event is None:
            return False
        logger.info("Cancelling generation for item %s", item_id)
        event.set()
        return True

    def check_cancellation(self, change: Change) -> None:
        """Check if a remote change is a cancellation request."""
        if not change.forward_component:
            return
        forward = change.forward_component
        request = forward.get("request")
        if not isinstance(request, dict):
            return
        if not request.get("cancelled_by_user"):
            return
        item_id = str(change.entity_id)
        self.cancel_generation(item_id)

    # --- Internal helpers ---

    def _build_request_params(
        self, request: dict[str, Any] | None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        spec = MODEL_SPECS.get(self.backend.current_model_key or "", {})
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
        gen_params, chat_template_params = self._build_request_params(item.request)
        batch = self.context.items_as_qwen_chat_messages()
        if not batch.messages:
            logger.warning(
                "No messages compiled for item=%s, skipping generation",
                getattr(item, "id", None),
            )
            return

        logger.debug(
            "Dispatching generation for item=%s stream=%s images=%d",
            getattr(item, "id", None),
            bool((item.request or {}).get("stream")),
            len(batch.images),
        )

        updated = cast(TextItem, item.copy())
        stream = bool((item.request or {}).get("stream"))
        if stream:
            await self._generate_stream(
                updated, batch.messages, batch.images, gen_params, chat_template_params
            )
        else:
            await self._generate_non_stream(
                updated, batch.messages, batch.images, gen_params, chat_template_params
            )

    async def _generate_non_stream(
        self,
        updated: TextItem,
        messages: list[dict[str, Any]],
        images: list,
        gen_params: dict[str, Any],
        chat_template_kwargs: dict[str, Any],
    ) -> None:
        try:
            result = await self.backend.generate(
                messages, images, gen_params, chat_template_kwargs
            )
            if result["status"] == "success":
                updated.text = result["text"]
                req = dict(updated.request or {})
                req["result"] = "success"
                req["completed"] = True
                updated.request = req
                updated.origin = "assistant"
                tokens_len = result.get("tokens")
                if tokens_len is not None:
                    meta = dict(updated.metadata or {})
                    meta["tokens_len"] = tokens_len
                    updated.metadata = meta
                logger.info(
                    "Non-stream generation complete item=%s text_len=%d tokens=%s",
                    getattr(updated, "id", None),
                    len(result["text"]),
                    tokens_len,
                )
            else:
                req = dict(updated.request or {})
                req["result"] = "error"
                req["error_message"] = result.get("error_message", "Unknown error")
                req["completed"] = True
                updated.request = req
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
        updated: TextItem,
        messages: list[dict[str, Any]],
        images: list,
        gen_params: dict[str, Any],
        chat_template_kwargs: dict[str, Any],
    ) -> None:
        uid = str(updated.id)
        cancel_event = threading.Event()
        self._cancel_events[uid] = cancel_event

        pieces: list[str] = []
        cancelled = False
        error: Exception | None = None

        try:
            async for chunk in self.backend.generate_stream(
                messages, images, gen_params, chat_template_kwargs, cancel_event
            ):
                if cancel_event.is_set():
                    cancelled = True
                    logger.info("Generation cancelled mid-stream for item %s", uid)
                    break
                if not chunk:
                    continue
                pieces.append(chunk)
                store: ContextStore = self.context._store
                store.text_append(str(updated.id), chunk)
                await asyncio.sleep(0)
        except Exception as exc:  # noqa: BLE001
            error = exc
            logger.error(
                "Inference failed (stream) for item %s: %s",
                getattr(updated, "id", None),
                exc,
                exc_info=True,
            )

        # Finalize
        full_text = "".join(pieces).strip()
        updated.text = full_text
        req = dict(updated.request or {})

        if error is not None:
            req["result"] = "error"
            req["error_message"] = str(error)
        elif cancelled:
            req["result"] = "cancelled"
            req["cancelled_by_user"] = True
        else:
            req["result"] = "success"

        req["completed"] = True
        updated.request = req
        updated.origin = "assistant"

        if not error and not cancelled and full_text:
            meta = dict(updated.metadata or {})
            # Approximate token count from text length (backends may not report)
            meta["text_len"] = len(full_text)
            updated.metadata = meta

        logger.info(
            "Stream generation done item=%s result=%s text_len=%d",
            uid,
            req["result"],
            len(full_text),
        )

        self._cancel_events.pop(uid, None)
        self.context.update(updated)
