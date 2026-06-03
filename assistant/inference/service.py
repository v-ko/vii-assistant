"""Inference service that reacts to context changes and produces model outputs."""

from __future__ import annotations

import asyncio
import json
import secrets
import threading
from typing import TYPE_CHECKING, Any, cast

import json_repair
from sivkit import get_logger
from sivkit.libs.model import load_from_dict
from sivkit.storage.change import Change

from assistant.inference.context import ContextItem, ContextManager, TextItem
from assistant.inference.context_store import ContextStore
from assistant.inference.focus_modes import (
    CLIENT_TOOLS,
    FOCUS_MODES,
    MAX_TOOL_CALL_DEPTH,
)
from assistant.model_configs import MODEL_SPECS

if TYPE_CHECKING:
    from assistant.inference.backend_protocol import InferenceBackend

logger = get_logger(__name__)

# Markers for Hermes-style tool calls (single tokens in vocab)
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"


async def generate_oneshot(
    context_manager: ContextManager,
    backend: "InferenceBackend",
    gen_params: dict[str, Any] | None = None,
    chat_template_params: dict[str, Any] | None = None,
    focus_mode: str | None = None,
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

    batch = context_manager.items_as_qwen_chat_messages(focus_mode=focus_mode)
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
        self._tool_call_depth: int = 0

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
            from sivkit.libs.model import dump_to_dict

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
        focus_mode = (item.metadata or {}).get("focus_mode") or (
            (item.request or {}).get("focus_mode")
        )
        if not focus_mode:
            raise ValueError(
                f"Generation request item {getattr(item, 'id', None)} "
                f"has no focus_mode in metadata or request"
            )
        batch = self.context.items_as_qwen_chat_messages(focus_mode=focus_mode)
        if not batch.messages:
            logger.warning(
                "No messages compiled for item=%s, skipping generation",
                getattr(item, "id", None),
            )
            return

        logger.debug(
            "Dispatching generation for item=%s stream=%s images=%d focus_mode=%s",
            getattr(item, "id", None),
            bool((item.request or {}).get("stream")),
            len(batch.images),
            focus_mode,
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
                full_text = result["text"]
                tokens_len = result.get("tokens")

                # Parse tool call from completed text
                text_before, tool_call_json = _extract_tool_call(full_text)

                updated.text = full_text
                req = dict(updated.request or {})
                req["result"] = "success"
                req["completed"] = True
                updated.request = req
                updated.origin = "assistant"
                meta = dict(updated.metadata or {})
                if tokens_len is not None:
                    meta["tokens_len"] = tokens_len
                if tool_call_json:
                    meta["tool_call"] = tool_call_json
                updated.metadata = meta

                logger.info(
                    "Non-stream generation complete item=%s text_len=%d tokens=%s tool_call=%s",
                    getattr(updated, "id", None),
                    len(updated.text),
                    tokens_len,
                    bool(tool_call_json),
                )
                self.context.update(updated)
                await self._handle_generation_result(
                    updated, text_before, tool_call_json
                )
            else:
                req = dict(updated.request or {})
                req["result"] = "error"
                req["error_message"] = result.get("error_message", "Unknown error")
                req["completed"] = True
                updated.request = req
                self.context.update(updated)
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
        tool_call_detected = False
        tool_call_pieces: list[str] = []

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

                # Tool call detection in stream
                if not tool_call_detected:
                    if TOOL_CALL_OPEN in chunk:
                        # Split: text before marker goes to output, rest starts accumulation
                        before, _, after = chunk.partition(TOOL_CALL_OPEN)
                        if before:
                            pieces.append(before)
                            store: ContextStore = self.context._store
                            store.text_append(str(updated.id), before)
                        tool_call_detected = True
                        if after:
                            tool_call_pieces.append(after)
                    else:
                        pieces.append(chunk)
                        store = self.context._store
                        store.text_append(str(updated.id), chunk)
                else:
                    # Accumulating tool call JSON
                    if TOOL_CALL_CLOSE in chunk:
                        before_close, _, _ = chunk.partition(TOOL_CALL_CLOSE)
                        if before_close:
                            tool_call_pieces.append(before_close)
                        # Done — stop reading the stream
                        break
                    else:
                        tool_call_pieces.append(chunk)

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
        if tool_call_detected:
            # Reconstruct full text preserving whitespace for token-exact replay
            raw_tool_json = "".join(tool_call_pieces)
            full_text_with_call = (
                full_text + TOOL_CALL_OPEN + raw_tool_json + TOOL_CALL_CLOSE
            )
            updated.text = full_text_with_call
        else:
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

        # Parse tool call if detected
        tool_call_json: dict[str, Any] | None = None
        if tool_call_detected and not error and not cancelled:
            raw_json = "".join(tool_call_pieces).strip()
            parsed = json_repair.loads(raw_json)
            if isinstance(parsed, dict) and "name" in parsed:
                tool_call_json = parsed
            else:
                logger.error(
                    "Failed to parse tool call JSON for item %s: %s", uid, raw_json
                )

        meta = dict(updated.metadata or {})
        if full_text:
            meta["text_len"] = len(full_text)
        if tool_call_json:
            meta["tool_call"] = tool_call_json
        updated.metadata = meta

        logger.info(
            "Stream generation done item=%s result=%s text_len=%d tool_call=%s",
            uid,
            req["result"],
            len(full_text),
            bool(tool_call_json),
        )

        self._cancel_events.pop(uid, None)
        self.context.update(updated)

        if not error and not cancelled:
            await self._handle_generation_result(updated, full_text, tool_call_json)

    # ------------------------------------------------------------------
    # Tool call dispatch
    # ------------------------------------------------------------------

    async def _handle_generation_result(
        self,
        updated: TextItem,
        text: str,
        tool_call_json: dict[str, Any] | None,
    ) -> None:
        """Shared handler called after both stream and non-stream generation.

        If a tool call was detected, dispatch to the target focus mode.
        If no tool call and this was a sub-mode (not main), inject result
        visible to caller and create a continuation request.
        """
        caller_mode = (updated.metadata or {}).get("caller_mode")

        if tool_call_json is None:
            # No tool call — generation is complete for this mode.
            if caller_mode and caller_mode != (updated.metadata or {}).get(
                "focus_mode"
            ):
                # This was a sub-mode (e.g., localization) completing.
                # Mark result visible to caller and continue the caller.
                meta = dict(updated.metadata or {})
                meta["visible_to"] = [caller_mode]
                updated.metadata = meta
                self.context.update(updated)
                await self._inject_continuation_request(caller_mode)
            else:
                # Main mode final response — done.
                self._tool_call_depth = 0
            return

        # Safety: max loop depth
        self._tool_call_depth += 1
        if self._tool_call_depth > MAX_TOOL_CALL_DEPTH:
            logger.warning(
                "Tool call depth %d exceeds max %d — stopping chain",
                self._tool_call_depth,
                MAX_TOOL_CALL_DEPTH,
            )
            self._tool_call_depth = 0
            return

        target_mode = tool_call_json.get("name", "")
        arguments = tool_call_json.get("arguments", {})

        # Determine the calling mode (from the generation request metadata)
        current_mode = (updated.metadata or {}).get("focus_mode", "main")

        if target_mode in CLIENT_TOOLS:
            # Client-side tool call (Python interpreter, click, scroll)
            self._inject_client_execution_request(
                updated, target_mode, arguments, current_mode
            )
            return

        mode_config = FOCUS_MODES.get(target_mode)
        if mode_config is None:
            logger.error("Unknown tool/focus mode in tool call: %s", target_mode)
            self._inject_tool_error(updated, f"Unknown tool/focus mode: {target_mode}")
            return

        # Server-side inference focus mode — dispatch directly
        instruction = arguments.get("instruction", "")
        await self._dispatch_focus_mode(updated, target_mode, instruction, current_mode)

    async def _dispatch_focus_mode(
        self,
        source: TextItem,
        target_mode: str,
        instruction: str,
        caller_mode: str,
    ) -> None:
        """Inject instruction + generation request for a server-side focus mode,
        then directly dispatch generation (bypass remote-change routing)."""
        ctx = self.context

        # Message: instruction for the target focus mode
        instruction_item = TextItem()
        instruction_item.position = ctx.next_position()
        instruction_item.origin = "user"
        instruction_item.text = instruction
        instruction_item.metadata = {
            "focus_mode": target_mode,
            "source_item_id": str(source.id),
        }
        ctx.insert(instruction_item)

        # Message: generation request for the target focus mode
        req_item = TextItem()
        req_item.position = ctx.next_position()
        req_item.origin = "assistant"
        req_item.request = {
            "stream": True,
            "focus_mode": target_mode,
        }
        req_item.metadata = {
            "focus_mode": target_mode,
            "caller_mode": caller_mode,
        }
        ctx.insert(req_item)

        logger.info(
            "Dispatching focus mode '%s' (server) from '%s', instruction_len=%d",
            target_mode,
            caller_mode,
            len(instruction),
        )

        # Directly dispatch generation (local item won't trigger handle_change)
        await self._dispatch_generation(req_item)

    async def _inject_continuation_request(self, target_mode: str) -> None:
        """Insert a generation request to continue the given mode, and dispatch it."""
        ctx = self.context
        req_item = TextItem()
        req_item.position = ctx.next_position()
        req_item.origin = "assistant"
        req_item.request = {
            "stream": True,
            "focus_mode": target_mode,
        }
        req_item.metadata = {"focus_mode": target_mode}
        ctx.insert(req_item)

        logger.info("Injected continuation request for mode '%s'", target_mode)

        # Directly dispatch (we're already inside the lock from the parent chain)
        await self._dispatch_generation(req_item)

    def _inject_client_execution_request(
        self,
        source: TextItem,
        target_mode: str,
        arguments: dict[str, Any],
        caller_mode: str,
    ) -> None:
        """Inject a request for client-side execution (Python interpreter or nav tools)."""
        ctx = self.context

        # Single item that signals client-side execution
        exec_item = TextItem()
        exec_item.position = ctx.next_position()
        exec_item.origin = "tool"
        exec_item.text = json.dumps(arguments)
        exec_item.request = {
            "execution": "client",
            "focus_mode": target_mode,
        }
        exec_item.metadata = {
            "focus_mode": target_mode,
            "caller_mode": caller_mode,
            "source_item_id": str(source.id),
            "arguments": arguments,
        }
        ctx.insert(exec_item)

        logger.info(
            "Dispatched client execution '%s' from '%s', args=%s",
            target_mode,
            caller_mode,
            list(arguments.keys()),
        )

    def _inject_tool_error(self, source: TextItem, error_message: str) -> None:
        """Inject an error result and continue the caller mode."""
        ctx = self.context
        caller_mode = (source.metadata or {}).get("focus_mode", "main")

        error_item = TextItem()
        error_item.position = ctx.next_position()
        error_item.origin = "tool"
        error_item.text = f"Error: {error_message}"
        error_item.metadata = {"visible_to": [caller_mode]}
        ctx.insert(error_item)

        # Continue the caller
        req_item = TextItem()
        req_item.position = ctx.next_position()
        req_item.origin = "assistant"
        req_item.request = {
            "stream": True,
            "focus_mode": caller_mode,
        }
        req_item.metadata = {"focus_mode": caller_mode}
        ctx.insert(req_item)

    def reset_tool_call_depth(self) -> None:
        """Reset tool call depth counter (e.g. on new user message)."""
        self._tool_call_depth = 0


def _extract_tool_call(text: str) -> tuple[str, dict[str, Any] | None]:
    """Extract tool call JSON from completed generation text.

    Returns (text_before_tool_call, parsed_json_or_None).
    """
    idx = text.find(TOOL_CALL_OPEN)
    if idx == -1:
        return text, None

    text_before = text[:idx].strip()
    after_open = text[idx + len(TOOL_CALL_OPEN) :]

    close_idx = after_open.find(TOOL_CALL_CLOSE)
    if close_idx == -1:
        # Malformed — no closing tag
        raw_json = after_open.strip()
    else:
        raw_json = after_open[:close_idx].strip()

    parsed = json_repair.loads(raw_json)
    if not isinstance(parsed, dict) or "name" not in parsed:
        logger.error("Failed to parse tool call JSON: %s", raw_json)
        return text, None

    return text_before, parsed
