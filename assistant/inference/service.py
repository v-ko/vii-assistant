"""Inference service that reacts to context changes and produces model outputs."""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import threading
from typing import TYPE_CHECKING, Any, cast

import json_repair
from sivkit.libs.model import load_from_dict
from sivkit.storage.change import Change

from assistant.inference.context import (
    ContextManager,
    ContextMessage,
    GateDecision,
    TextMessage,
    gate_state,
)
from assistant.inference.context_store import ContextStore
from assistant.inference.focus_modes import (
    CLIENT_TOOLS,
    FOCUS_MODES,
    MAX_TOOL_CALL_DEPTH,
)
from assistant.model_configs import MODEL_SPECS

if TYPE_CHECKING:
    from assistant.inference.backend_protocol import InferenceBackend

logger = logging.getLogger(__name__)

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
        # Held supervised turns already resumed (dedup against duplicate verdict
        # deliveries spawning the continuation twice).
        self._resumed_turns: set[str] = set()
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
        if not isinstance(item, ContextMessage):
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

    def cancel_all_generations(self) -> None:
        """Break every in-flight generation stream (used on chain cancellation)."""
        for item_id, event in list(self._cancel_events.items()):
            logger.info("Cancelling generation for item %s (cancel-all)", item_id)
            event.set()

    def _chain_cancelled(self, item: ContextMessage | None) -> bool:
        """True if ``item`` or any of its previous_message_id ancestors is cancelled."""
        seen: set[str] = set()
        cur = item
        while cur is not None:
            cid = str(getattr(cur, "id", ""))
            if not cid or cid in seen:
                return False
            seen.add(cid)
            if getattr(cur, "cancelled", False):
                return True
            pid = getattr(cur, "previous_message_id", "") or ""
            if not pid:
                return False
            cur = self.context._store.find_one(id=pid)
        return False

    def check_cancellation(self, change: Change) -> None:
        """React to a client cancellation (top-level ``cancelled`` flag): abort
        all in-flight generations. Birth suppression is handled by the ancestry
        guard at each continuation site."""
        if not change.forward_component:
            return
        if not change.forward_component.get("cancelled"):
            return
        self.cancel_all_generations()

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

    async def _dispatch_generation(self, item: ContextMessage) -> None:
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

        updated = cast(TextMessage, item.copy())
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
        updated: TextMessage,
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
                updated.size = tokens_len or 0
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
                live = self.context._store.find_one(id=str(updated.id))
                if live is not None:
                    updated.cancelled = live.cancelled
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
                live = self.context._store.find_one(id=str(updated.id))
                if live is not None:
                    updated.cancelled = live.cancelled
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
            live = self.context._store.find_one(id=str(updated.id))
            if live is not None:
                updated.cancelled = live.cancelled
            self.context.update(updated)

    async def _generate_stream(
        self,
        updated: TextMessage,
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

                # Stream every chunk live, including any raw <tool_call> markup.
                # The tool call is extracted from the finalized text below, which
                # is tokenization-independent (unlike per-chunk marker matching,
                # which breaks when <tool_call> is not a single vocab token, e.g.
                # for Gemma).
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

        # Finalize: detect the tool call on the complete text. When present,
        # collapse the message to the text before <tool_call> and drop the call
        # markup plus anything after it (one tool call per turn).
        raw_full_text = "".join(pieces)
        tool_call_json: dict[str, Any] | None = None
        if not error and not cancelled:
            full_text, tool_call_json = _extract_tool_call(raw_full_text)
        else:
            full_text = raw_full_text.strip()
        updated.text = full_text
        req = dict(updated.request or {})

        if error is not None:
            req["result"] = "error"
            req["error_message"] = str(error)
        elif cancelled:
            req["result"] = "cancelled"
        else:
            req["result"] = "success"

        req["completed"] = True
        updated.request = req
        updated.origin = "assistant"
        counter = getattr(self.backend, "count_tokens", None)
        updated.size = counter(updated.text) if counter is not None else 0

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
        # Don't revert a cancellation the client may have set while we streamed.
        live = self.context._store.find_one(id=uid)
        if live is not None:
            updated.cancelled = live.cancelled
        self.context.update(updated)

        if not error and not cancelled:
            await self._handle_generation_result(updated, full_text, tool_call_json)

    # ------------------------------------------------------------------
    # Tool call dispatch
    # ------------------------------------------------------------------

    def _supervision_gate(self, updated: TextMessage) -> GateDecision:
        """Read the live supervision verdict for a completed turn.

        Reloads from the store so a verdict that landed while we finalized is
        seen. HOLD = await verdict (withhold continuation); SUPPRESS = error
        (server never resumes); CONTINUE = proceed.
        """
        live = self.context._store.find_one(id=str(updated.id))
        return gate_state(live if live is not None else updated)

    async def _resume_held_turn(self, item: TextMessage) -> None:
        """Re-run a turn's withheld continuation after a pass/correct verdict.

        Stateless: the continuation is re-derived from the turn's stored
        ``metadata["tool_call"]`` (no parked closures), so it survives reconnects.
        """
        tid = str(item.id)
        if tid in self._resumed_turns:
            return
        self._resumed_turns.add(tid)
        tool_call_json = (item.metadata or {}).get("tool_call")
        async with self._lock:
            await self._handle_generation_result(item, "", tool_call_json)

    def resume_target(self, change: Change) -> TextMessage | None:
        """If a remote change is a pass/correct verdict on a held turn, return
        the live turn item to resume; else None."""
        if change.is_delete() or not change.forward_component:
            return None
        if change.forward_component.get("teacher_feedback") not in ("pass", "correct"):
            return None
        item = self.context._store.find_one(id=str(change.entity_id))
        return item if isinstance(item, TextMessage) else None

    @staticmethod
    def _arm_supervision(source: ContextMessage, new_item: TextMessage) -> None:
        """Re-arm a freshly-born continuation request for review if the chain is
        supervised (carried by the source turn's non-empty teacher_feedback)."""
        if getattr(source, "teacher_feedback", ""):
            new_item.teacher_feedback = "pending"

    async def _handle_generation_result(
        self,
        updated: TextMessage,
        text: str,
        tool_call_json: dict[str, Any] | None,
    ) -> None:
        """Shared handler called after both stream and non-stream generation.

        If a tool call was detected, dispatch to the target focus mode.
        If no tool call and this was a sub-mode (not main), inject result
        visible to caller and create a continuation request.
        """
        # Supervision gate: in supervised mode the turn is born "pending". Hold
        # the continuation until the client writes a verdict; the resume path
        # re-enters this method once it flips to pass/correct.
        decision = self._supervision_gate(updated)
        if decision is GateDecision.HOLD:
            logger.info("Turn %s held for supervision", getattr(updated, "id", None))
            return
        if decision is GateDecision.SUPPRESS:
            logger.info(
                "Turn %s suppressed by supervisor", getattr(updated, "id", None)
            )
            self._tool_call_depth = 0
            return

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
                live = self.context._store.find_one(id=str(updated.id))
                if live is not None:
                    updated.cancelled = live.cancelled
                self.context.update(updated)
                await self._inject_continuation_request(caller_mode, updated)
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

        tool_name = tool_call_json.get("name", "")
        arguments = tool_call_json.get("arguments", {})

        # The `focus` tool delegates to a server-side focus mode named by its
        # `mode` argument. Other tool names map directly to client tools.
        if tool_name == "focus":
            target_mode = arguments.get("mode", "")
        else:
            target_mode = tool_name

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
            await self._inject_tool_error(
                updated, f"Unknown tool/focus mode: {target_mode}"
            )
            return

        # Server-side inference focus mode — dispatch directly
        instruction = arguments.get("instruction", "")
        await self._dispatch_focus_mode(updated, target_mode, instruction, current_mode)

    async def _dispatch_focus_mode(
        self,
        source: TextMessage,
        target_mode: str,
        instruction: str,
        caller_mode: str,
    ) -> None:
        """Inject instruction + generation request for a server-side focus mode,
        then directly dispatch generation (bypass remote-change routing)."""
        ctx = self.context

        if self._chain_cancelled(source):
            logger.info(
                "Chain cancelled — not dispatching focus mode '%s'", target_mode
            )
            return

        # Message: instruction for the target focus mode
        instruction_item = TextMessage()
        instruction_item.position = ctx.next_position()
        instruction_item.origin = "user"
        instruction_item.text = instruction
        instruction_item.previous_message_id = str(source.id)
        instruction_item.metadata = {
            "focus_mode": target_mode,
            "source_item_id": str(source.id),
        }
        ctx.insert(instruction_item)

        # Message: generation request for the target focus mode
        req_item = TextMessage()
        req_item.position = ctx.next_position()
        req_item.origin = "assistant"
        req_item.previous_message_id = str(instruction_item.id)
        req_item.request = {
            "stream": True,
            "focus_mode": target_mode,
        }
        req_item.metadata = {
            "focus_mode": target_mode,
            "caller_mode": caller_mode,
        }
        self._arm_supervision(source, req_item)
        ctx.insert(req_item)

        logger.info(
            "Dispatching focus mode '%s' (server) from '%s', instruction_len=%d",
            target_mode,
            caller_mode,
            len(instruction),
        )

        # Directly dispatch generation (local item won't trigger handle_change)
        await self._dispatch_generation(req_item)

    async def _inject_continuation_request(
        self, target_mode: str, source: TextMessage
    ) -> None:
        """Insert a generation request to continue the given mode, and dispatch it."""
        ctx = self.context
        if self._chain_cancelled(source):
            logger.info(
                "Chain cancelled — not injecting continuation for '%s'", target_mode
            )
            return
        req_item = TextMessage()
        req_item.position = ctx.next_position()
        req_item.origin = "assistant"
        req_item.previous_message_id = str(source.id)
        req_item.request = {
            "stream": True,
            "focus_mode": target_mode,
        }
        req_item.metadata = {"focus_mode": target_mode}
        self._arm_supervision(source, req_item)
        ctx.insert(req_item)

        logger.info("Injected continuation request for mode '%s'", target_mode)

        # Directly dispatch (we're already inside the lock from the parent chain)
        await self._dispatch_generation(req_item)

    def _inject_client_execution_request(
        self,
        source: TextMessage,
        target_mode: str,
        arguments: dict[str, Any],
        caller_mode: str,
    ) -> None:
        """Inject a request for client-side execution (Python interpreter or nav tools)."""
        ctx = self.context

        # Single item that signals client-side execution
        exec_item = TextMessage()
        exec_item.position = ctx.next_position()
        exec_item.origin = "tool"
        exec_item.text = json.dumps(arguments)
        exec_item.previous_message_id = str(source.id)
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

    async def _inject_tool_error(self, source: TextMessage, error_message: str) -> None:
        """Inject an error result and continue the caller mode."""
        ctx = self.context
        caller_mode = (source.metadata or {}).get("focus_mode", "main")

        if self._chain_cancelled(source):
            logger.info("Chain cancelled — not injecting tool error continuation")
            return

        error_item = TextMessage()
        error_item.position = ctx.next_position()
        error_item.origin = "tool"
        error_item.text = f"Error: {error_message}"
        error_item.previous_message_id = str(source.id)
        error_item.metadata = {"visible_to": [caller_mode]}
        ctx.insert(error_item)

        # Continue the caller
        req_item = TextMessage()
        req_item.position = ctx.next_position()
        req_item.origin = "assistant"
        req_item.previous_message_id = str(error_item.id)
        req_item.request = {
            "stream": True,
            "focus_mode": caller_mode,
        }
        req_item.metadata = {"focus_mode": caller_mode}
        self._arm_supervision(source, req_item)
        ctx.insert(req_item)

        # Directly dispatch (we're already inside the lock from the parent chain)
        await self._dispatch_generation(req_item)

    def reset_tool_call_depth(self) -> None:
        """Reset tool call depth counter (e.g. on new user message)."""
        self._tool_call_depth = 0


def _loads_tool_json(raw_json: str) -> Any:
    """Parse tool-call JSON, tolerating literal newlines in code strings.

    json.loads(strict=False) accepts raw control chars (which models emit
    inside multi-line `code`); json_repair is a fallback for other glitches.
    """
    try:
        return json.loads(raw_json, strict=False)
    except json.JSONDecodeError:
        return json_repair.loads(raw_json)


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

    parsed = _loads_tool_json(raw_json)
    if not isinstance(parsed, dict) or "name" not in parsed:
        logger.error("Failed to parse tool call JSON: %s", raw_json)
        return text, None

    return text_before, parsed
