from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from PIL import Image
from PySide6.QtGui import QGuiApplication
from sivkit.storage.change import Change
from sivkit.storage.delta import Delta
from sivkit.util.rectangle import Rectangle

from assistant.facade import vii
from assistant.image_ops import (
    qwen_grid_to_original_image,
    qwen_point_to_original_image,
    resize_to_target,
    scale_qwen_bbox_xyxy,
    scale_qwen_point,
)
from assistant.inference.context import (
    ContextMessage,
    ImageMessage,
    TextMessage,
)
from assistant.inference.context_store import OP_SEP
from assistant.inference.focus_modes import PERCEPTION_MODES
from assistant.model_configs import get_resolution_for_model
from assistant.overlay_actions import (
    clear_supervised_review,
    show_supervised_review,
)
from assistant.services.overlay_manager import OverlayManager
from assistant.services.segment_parsing import (
    hfi,
    parse_segment_output,
)
from assistant.services.supervised_gate import SupervisedGate
from assistant.util import Shape, get_screen_by_name
from assistant.utils.capture_utils import take_screenshot
from assistant.utils.image_utils import qpixmap_to_pil
from assistant.view_states.overlay import OverlayMode
from assistant.view_states.settings import ExecutionMode

log = logging.getLogger(__name__)


class ChainStopReason(Enum):
    """Why an agent chain (a run_chain invocation) ended."""

    TURN_COMPLETED = "turn_completed"  # assistant produced no further tool call
    MAX_TURNS = "max_turns"  # hit MAX_TURNS guard
    STOPPED = "stopped"  # cancel_chain() called (user/experiment stop)
    DISCONNECTED = "disconnected"  # websocket dropped mid-chain
    ERROR = "error"  # unexpected failure


@dataclass(frozen=True)
class ChainResult:
    """Lifecycle-only result of an agent chain. No data payload —

    consumers read final state from the context store and interpreter.
    """

    stop_reason: ChainStopReason
    turn_count: int


class ChainError(RuntimeError):
    """Raised into a chain awaiter for abnormal endings (stopped/error)."""

    def __init__(self, reason: ChainStopReason, message: str = "") -> None:
        super().__init__(message or reason.value)
        self.reason = reason


class HybridSegmentService:
    """Watches context changes for overlay updates and agent tool execution.

    Pure parsing/extraction lives in segment_parsing module.
    This class owns the agent chat-driving loop: it inserts generation
    requests (run_chain), drives tool-call continuation, and signals when a
    chain settles (no further tool calls) or is aborted.
    """

    MAX_TURNS = 5

    def __init__(self) -> None:
        # Dedup for client-side execution requests (one execution per item)
        self._processed_tool_call_ids: set[str] = set()
        self._turn_count: int = 0
        self._stopped: bool = False
        # Non-reentrant guard — only one processing chain at a time
        self._processing: bool = False
        # Active chain settle future (resolved on TURN_COMPLETED/MAX_TURNS,
        # failed on STOPPED/DISCONNECTED/ERROR). None when no driven chain.
        self._chain_future: asyncio.Future[ChainResult] | None = None
        # Focus mode whose top-level completion settles the active chain.
        self._active_focus_mode: str = "main"
        # Last known pointer position (screen coords) for scroll
        self._pointer_x: int | None = None
        self._pointer_y: int | None = None
        # Supervised gate — tracks the turn awaiting a verdict and writes the
        # verdict to the store (which syncs to the server and resumes the chain).
        self.supervised_gate = SupervisedGate()
        # Guard overlays for AUTO mode
        self.overlay_manager = OverlayManager()

    @property
    def context_manager(self):
        return vii.project_manager.context_manager

    def _supervised(self) -> bool:
        """True if the user has supervised execution mode enabled."""
        return (
            vii.app.view_state.settings_VS.execution_mode
            == ExecutionMode.SUPERVISED.value
        )

    def reset_turns(self) -> None:
        """Reset all per-conversation state for a fresh session/sample."""
        self._turn_count = 0
        self._stopped = False
        self._processing = False
        self._active_focus_mode = "main"
        self._chain_future = None
        self._processed_tool_call_ids.clear()
        self.supervised_gate.clear()

    # ------------------------------------------------------------------
    # Chain ownership: insert requests, drive continuation, signal settle
    # ------------------------------------------------------------------

    def run_chain(
        self, focus_mode: str = "main", *, request_extras: dict | None = None
    ) -> asyncio.Future[ChainResult]:
        """Start an agent chain in ``focus_mode`` and return its settle future.

        The service owns the request item: it is inserted here (never by the
        caller). The future resolves with a ChainResult on natural completion
        (TURN_COMPLETED) or MAX_TURNS, and fails with ChainError/ConnectionError
        on stop/disconnect/error.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

        self._stopped = False
        self._turn_count = 0
        self._active_focus_mode = focus_mode
        fut: asyncio.Future[ChainResult] = loop.create_future()
        self._chain_future = fut
        # Retrieve the exception even if nobody awaits (live mode), so a failed
        # chain does not emit "future exception was never retrieved" warnings.
        fut.add_done_callback(lambda f: f.cancelled() or f.exception())

        ctx = self.context_manager
        req_item = TextMessage()
        req_item.position = ctx.next_position()
        req_item.origin = "assistant"
        request: dict[str, Any] = {"stream": True, "focus_mode": focus_mode}
        if request_extras:
            request.update(request_extras)
        req_item.request = request
        req_item.metadata = {"focus_mode": focus_mode}
        ctx.insert(req_item)

        vii.app.view_state.settings_VS.assistant_working = True
        return fut

    def cancel_chain(self, reason: str = "cancelled") -> None:
        """Stop the active chain: halt continuation and fail the settle future."""
        self._stopped = True
        self.supervised_gate.clear()
        clear_supervised_review()
        self._settle_chain(ChainStopReason.STOPPED, message=reason)

    def fail_chain_disconnected(self) -> None:
        """Fail the active chain because the websocket dropped mid-chain."""
        self._stopped = True
        self._settle_chain(ChainStopReason.DISCONNECTED)

    def _settle_chain(self, reason: ChainStopReason, *, message: str = "") -> None:
        """Resolve/fail the active chain future and update UI working state."""
        settings_VS = vii.app.view_state.settings_VS
        settings_VS.assistant_working = False
        # Fire-and-forget broadcast for incidental listeners (UI, logging).
        # Emitted in both driven (run_chain) and live modes, regardless of
        # whether anyone is awaiting the settle future.
        settings_VS.chain_ended.emit(reason.value)
        fut = self._chain_future
        self._chain_future = None
        if fut is None or fut.done():
            return
        if reason in (ChainStopReason.TURN_COMPLETED, ChainStopReason.MAX_TURNS):
            fut.set_result(ChainResult(reason, self._turn_count))
        elif reason is ChainStopReason.DISCONNECTED:
            fut.set_exception(
                ConnectionError("WebSocket disconnected during inference")
            )
        else:
            fut.set_exception(ChainError(reason, message))

    def handle_completed_message(self, item: TextMessage) -> None:
        """Detect chain settle: a completed top-level assistant message with no
        tool call means the agent handed control back to the user."""
        if not isinstance(item.request, dict):
            return
        if item.origin != "assistant":
            return
        if getattr(item, "cancelled", False):
            return
        meta = item.metadata or {}
        if meta.get("tool_call"):
            # A tool call is pending — the chain continues (server injects an
            # execution=client item or dispatches a sub-mode).
            return
        # No tool call. Only settle on the top-level mode (sub-modes like
        # localization continue their caller via a server-injected request).
        if meta.get("focus_mode", "main") != self._active_focus_mode:
            return
        self._settle_chain(ChainStopReason.TURN_COMPLETED)

    def changed_context_items(self, delta: Delta) -> list[ContextMessage]:
        items: list[ContextMessage] = []
        seen: set[str] = set()
        ctx_mgr = self.context_manager

        for key, change_data in delta.asdict().items():
            if OP_SEP in key:
                entity_id = key.split(OP_SEP, 1)[0]
            else:
                change = Change(*change_data)
                if change.is_delete():
                    continue
                entity_id = change.entity_id

            if entity_id in seen:
                continue
            seen.add(entity_id)
            entity = ctx_mgr._store.find_one(id=entity_id)
            if isinstance(entity, ContextMessage):
                items.append(entity)
        return items

    def _chain_cancelled(self, item_id: str) -> bool:
        """True if the item or any of its previous_message_id ancestors is cancelled."""
        store = self.context_manager._store
        seen: set[str] = set()
        cur = store.find_one(id=item_id) if item_id else None
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
            cur = store.find_one(id=pid)
        return False

    # ------------------------------------------------------------------
    # Focus-mode client-side execution (tool call dispatch)
    # ------------------------------------------------------------------

    async def process_client_execution_request(self, item: TextMessage) -> None:
        """Handle a client-side execution request dispatched by InferenceService.

        Supports: python (interpreter), click_at, scroll.
        """
        item_id = str(item.id)
        if item_id in self._processed_tool_call_ids:
            return
        self._processed_tool_call_ids.add(item_id)

        focus_mode = (item.request or {}).get("focus_mode", "")
        arguments = (item.metadata or {}).get("arguments", {})
        caller_mode = (item.metadata or {}).get("caller_mode", "main")

        # Action tools that move the real pointer / world. In supervised mode
        # these were already approved at the producing turn's gate (completion
        # time), so by the time this execution request exists it is cleared.
        is_action = focus_mode in ("click_at", "scroll")
        in_experiment = vii.app.view_state.overlay_VS.mode == OverlayMode.EXPERIMENT
        if is_action and in_experiment:
            # Grounding experiments are static single-image tasks: never move the
            # real pointer. Record the attempt and continue the chain.
            self._insert_result_and_continue(
                f"{focus_mode} skipped (experiment mode)",
                item_id,
                caller_mode,
                is_action=False,
            )
            return

        # Show preview shapes on overlay before execution. Combine shapes
        # parsed from the current item (only the last message) with shapes
        # read from the persistent interpreter state (e.g. target['bbox']).
        from assistant.services.segment_parsing import extract_shapes_from_item

        preview_shapes = (
            extract_shapes_from_item(item) + self._extract_shapes_from_interpreter()
        )
        if preview_shapes:
            self._show_preview_shapes(preview_shapes)

        if focus_mode == "python":
            result_text = self._execute_python_code(arguments.get("code", ""))
        elif focus_mode == "click_at":
            result_text = await self._exec_click_at_coords(
                arguments.get("x", 0), arguments.get("y", 0)
            )
        elif focus_mode == "scroll":
            result_text = await self._exec_scroll(
                arguments.get("coordinate", [500, 500]),
                arguments.get("direction", "down"),
                arguments.get("amount", 3),
            )
        else:
            result_text = f"Error: unknown client execution mode '{focus_mode}'"

        self._insert_result_and_continue(result_text, item_id, caller_mode, is_action)

    def _insert_result_and_continue(
        self, result_text: str, source_item_id: str, caller_mode: str, is_action: bool
    ) -> None:
        """Insert tool result, optionally capture screen, and continue caller mode."""
        ctx = self.context_manager

        # If the chain was cancelled mid-execution, record the result but do
        # not drive another turn. This is checked both via the local stopped
        # flag and via the synced lineage (so a cancellation that arrived from
        # the peer / survived a reconnect is also honoured).
        if self._stopped or self._chain_cancelled(source_item_id):
            log.info("chain stopped — not continuing after client execution")
            return

        # Insert result visible to caller mode
        result_item = TextMessage()
        result_item.position = ctx.next_position()
        result_item.origin = "tool"
        result_item.text = result_text
        result_item.previous_message_id = source_item_id
        result_item.metadata = {
            "visible_to": [caller_mode],
            "source_item_id": source_item_id,
        }
        ctx.insert(result_item)
        tail_id = str(result_item.id)

        # Insert a fresh observation (screenshot) ONLY after input/action tools.
        # Read-only tools (python/crop_image/curator_push) must not trigger one.
        if is_action:
            self._insert_screen_capture()

        # Check loop limit
        self._turn_count += 1
        if self._turn_count >= self.MAX_TURNS:
            log.warning(
                "max_turns=%d reached in client execution, stopping", self.MAX_TURNS
            )
            self._settle_chain(ChainStopReason.MAX_TURNS)
            return

        # Continue the caller mode
        req_item = TextMessage()
        req_item.position = ctx.next_position()
        req_item.origin = "assistant"
        req_item.previous_message_id = tail_id
        req_item.request = {
            "stream": True,
            "focus_mode": caller_mode,
        }
        req_item.metadata = {"focus_mode": caller_mode}
        # Re-arm for review: in supervised mode every continuation turn is gated.
        if self._supervised():
            req_item.teacher_feedback = "pending"
        ctx.insert(req_item)

        # log.info(
        #     "Client execution complete, continuing '%s' (turn %d/%d)",
        #     caller_mode,
        #     self._turn_count,
        #     self.MAX_TURNS,
        # )

    def _execute_python_code(self, code: str) -> str:
        """Execute Python code in the persistent interpreter namespace."""
        try:
            exec_globals = dict(hfi.variables)
            for name, fn_def in hfi._functions.items():
                exec_globals[name] = fn_def.func

            # Override stubs with real implementations
            exec_globals["curator_push"] = self._curator_push_sync
            exec_globals["crop_image"] = self._crop_image_sync

            exec(code, exec_globals)  # noqa: S102

            # Update hfi variables with any new assignments
            for key, val in exec_globals.items():
                if key.startswith("_"):
                    continue
                if key in hfi._functions:
                    continue
                if key in ("curator_push", "crop_image"):
                    continue
                hfi.variables[key] = val

            return "Tool 'python': no error"
        except Exception as exc:  # noqa: BLE001
            log.error("Python execution error: %s", exc, exc_info=True)
            return f"Error: {exc}"

    def _curator_push_sync(
        self, feed: str, content: dict, metadata: dict | None = None
    ) -> str:
        """Synchronous curator push for use inside exec'd Python code."""
        import httpx as _httpx

        from assistant.services.curator_client import CURATOR_SERVER_URL

        payload: dict = {"feed": feed, "content": content}
        if metadata:
            payload["metadata"] = metadata
        try:
            resp = _httpx.post(f"{CURATOR_SERVER_URL}/push", json=payload, timeout=10)
            resp.raise_for_status()
            return "ok"
        except _httpx.ConnectError:
            raise RuntimeError("curator server unreachable") from None
        except _httpx.HTTPStatusError as exc:
            raise RuntimeError(f"curator error: {exc.response.text.strip()}") from None

    def _crop_image_sync(self, bbox: list) -> str:
        """Crop the latest screenshot at bbox (0-1000 grid). Returns b64 PNG."""
        screenshot_b64 = self._get_last_screenshot_b64()
        if screenshot_b64 is None:
            raise RuntimeError("no screenshot available to crop")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise RuntimeError(f"bbox must be [x1, y1, x2, y2], got {bbox!r}")
        bbox_tuple = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        b64, _w, _h = self._crop_screenshot(screenshot_b64, bbox_tuple)
        return b64

    async def _exec_click_at_coords(self, x: int, y: int) -> str:
        """Click at coordinates (0-1000 grid). Converts to screen pixels."""
        from assistant.services import input_control

        ox, oy, screen_w, screen_h = self._resolve_screen_geometry()
        if screen_w == 0 or screen_h == 0:
            return "Error: cannot determine screen resolution"
        px = ox + int(x / 1000 * screen_w)
        py = oy + int(y / 1000 * screen_h)
        if not await input_control.click(px, py):
            return f"Error: click at ({px}, {py}) failed"
        self._pointer_x = px
        self._pointer_y = py
        return f"clicked at grid ({x}, {y}) → screen ({px}, {py})"

    def update_overlay_from_text(self, text: str) -> None:
        if not text.strip():
            return
        output_model = parse_segment_output(text)
        rects: list[Rectangle] = []
        # Collect all bboxes
        bbox_list = output_model.get("bbox", [])
        if bbox_list:
            rects.extend(bbox_list)
        # Collect all image regions (treat as rectangles for now)
        img_list = output_model.get("image", [])
        for img_val in img_list:
            if isinstance(img_val, Rectangle):
                rects.append(img_val)
        points = output_model.get("points", [])
        # log.info(
        #     f"Extracted {len(rects)} rectangles and {len(points)} points from AI output"
        # )
        if not rects and not points:
            # No shapes in this message's text. Fall back to shapes held in the
            # persistent interpreter state (e.g. target['bbox']) so a located
            # box stays visible across subsequent messages instead of being
            # cleared by every bbox-less message.
            interp_shapes = self._extract_shapes_from_interpreter()
            if interp_shapes:
                self._show_preview_shapes(interp_shapes)
            else:
                vii.app.view_state.overlay_VS.shapes = []
        else:
            output_w, output_h = self._resolve_output_resolution()
            input_w, input_h = self._resolve_input_resolution(output_w, output_h)
            shapes = self._convert_to_shapes(
                rects,
                points,
                input_w=input_w,
                input_h=input_h,
                output_w=output_w,
                output_h=output_h,
            )
            vii.app.view_state.overlay_VS.shapes = shapes

    def _resolve_output_resolution(self) -> tuple[int, int]:
        """Get the target screen resolution for overlay rendering."""
        _, _, w, h = self._resolve_screen_geometry()
        return w, h

    def _resolve_screen_geometry(self) -> tuple[int, int, int, int]:
        """Get (x_offset, y_offset, width, height) of the capture screen."""
        capture = vii.app.view_state.capture_screen_info
        if capture:
            screen = get_screen_by_name(capture.name)
        else:
            screen = QGuiApplication.primaryScreen()
        if not screen:
            log.error("No screen available for shape conversion")
            return 0, 0, 0, 0
        geo = screen.geometry()
        x, y = int(geo.x()), int(geo.y())
        w, h = int(geo.width()), int(geo.height())
        return x, y, w, h

    def _resolve_input_resolution(
        self, fallback_w: int, fallback_h: int
    ) -> tuple[int, int]:
        """Get the resolution the model saw — from the last image item or model config."""
        for it in self.context_manager.items_reversed():
            if isinstance(it, ImageMessage):
                return it.width, it.height

        model_key = vii.get_config().selected_model
        w, h = get_resolution_for_model(model_key)
        return w, h

    def _convert_to_shapes(
        self,
        rects: list[Rectangle],
        points: list[tuple[int, int]] | None = None,
        *,
        input_w: int,
        input_h: int,
        output_w: int,
        output_h: int,
    ) -> list[Shape]:
        shapes: list[Shape] = []

        # In experiment mode, map through resize_meta + display transform
        overlay_vs = vii.app.view_state.overlay_VS
        use_display_transform = (
            overlay_vs.mode == OverlayMode.EXPERIMENT
            and overlay_vs.resize_meta is not None
        )
        dt = overlay_vs.display_transform if use_display_transform else None
        meta = overlay_vs.resize_meta if use_display_transform else None

        for r in rects:
            x1 = r.x()
            y1 = r.y()
            x2 = r.right()
            y2 = r.bottom()

            if dt is not None and meta is not None:
                # Model 0-1000 grid → original image pixels (undo padding+scale)
                ox1, oy1, ox2, oy2 = qwen_grid_to_original_image(
                    (int(x1), int(y1), int(x2), int(y2)), meta
                )
                # Original image pixels → widget pixels via display transform
                wx1, wy1 = dt.image_to_widget(ox1, oy1)
                wx2, wy2 = dt.image_to_widget(ox2, oy2)
                sx, sy = int(wx1), int(wy1)
                w = max(0, int(wx2) - sx)
                h = max(0, int(wy2) - sy)
            else:
                sx, sy, ex, ey = scale_qwen_bbox_xyxy(
                    (int(x1), int(y1), int(x2), int(y2)),
                    input_w=input_w,
                    input_h=input_h,
                    orig_w=output_w,
                    orig_h=output_h,
                    coords_are_qwen_grid=True,
                )
                w = max(0, ex - sx)
                h = max(0, ey - sy)

            rect_shape: Shape = {
                "type": "rect",
                "geometry": (sx, sy, w, h),
                "color": "#FF0000",
            }
            shapes.append(rect_shape)

        for pt in points or []:
            if dt is not None and meta is not None:
                ox, oy = qwen_point_to_original_image(pt, meta)
                wx, wy = dt.image_to_widget(ox, oy)
                sx, sy = int(wx), int(wy)
            else:
                sx, sy = scale_qwen_point(
                    pt,
                    input_w=input_w,
                    input_h=input_h,
                    orig_w=output_w,
                    orig_h=output_h,
                    coords_are_qwen_grid=True,
                )
            point_shape: Shape = {
                "type": "point",
                "geometry": (int(sx), int(sy)),
                "color": "#00FF00",
            }
            shapes.append(point_shape)

        return shapes

    def _show_preview_shapes(self, grid_shapes: list) -> None:
        """Convert shapes in Qwen 0-1000 grid to screen coords and show on overlay."""
        from assistant.util import Shape

        output_w, output_h = self._resolve_output_resolution()
        input_w, input_h = self._resolve_input_resolution(output_w, output_h)

        screen_shapes: list[Shape] = []
        for shape in grid_shapes:
            if shape["type"] == "rect":
                x, y, w, h = shape["geometry"]
                # Convert from xywh in grid to screen pixels
                from assistant.image_ops import scale_qwen_bbox_xyxy

                sx, sy, ex, ey = scale_qwen_bbox_xyxy(
                    (x, y, x + w, y + h),
                    input_w=input_w,
                    input_h=input_h,
                    orig_w=output_w,
                    orig_h=output_h,
                    coords_are_qwen_grid=True,
                )
                sw = max(0, ex - sx)
                sh = max(0, ey - sy)
                screen_shapes.append(
                    {"type": "rect", "geometry": (sx, sy, sw, sh), "color": "#FFAA00"}
                )
            elif shape["type"] == "point":
                x, y = shape["geometry"]
                from assistant.image_ops import scale_qwen_point

                sx, sy = scale_qwen_point(
                    (x, y),
                    input_w=input_w,
                    input_h=input_h,
                    orig_w=output_w,
                    orig_h=output_h,
                    coords_are_qwen_grid=True,
                )
                screen_shapes.append(
                    {"type": "point", "geometry": (sx, sy), "color": "#00FF00"}
                )

        vii.app.view_state.overlay_VS.shapes = screen_shapes

    def _extract_shapes_from_interpreter(self) -> list[Shape]:
        """Extract preview shapes from the persistent interpreter state.

        Reads ``hfi.variables["target"]["bbox"]`` (xyxy, Qwen 0-1000 grid),
        with a fallback to a top-level ``bbox`` variable, mirroring
        experiments_manager._extract_bbox_from_hfi. Returns rect shapes in
        xywh grid coords (the format _show_preview_shapes expects).
        """
        shapes: list[Shape] = []

        candidates: list = []
        target = hfi.variables.get("target")
        if isinstance(target, dict):
            candidates.append(target.get("bbox"))
        candidates.append(hfi.variables.get("bbox"))

        for bbox in candidates:
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                continue
            try:
                x1, y1, x2, y2 = (int(v) for v in bbox)
            except (ValueError, TypeError):
                continue
            w = x2 - x1
            h = y2 - y1
            if w > 0 and h > 0:
                shapes.append({"type": "rect", "geometry": (x1, y1, w, h)})

        return shapes

    # ------------------------------------------------------------------
    # Supervised mode gate
    # ------------------------------------------------------------------

    async def review_completed_turn(self, item: TextMessage) -> None:
        """Supervised mode: gate/label a completed turn, then settle the chain
        on a top-level completion.

        Non-input turns auto-pass (label only, no window); input actions
        (click_at/scroll) prompt the supervisor. The verdict is written to the
        store, which syncs to the server: pass/correct resumes the withheld
        continuation, error hard-suppresses it.
        """
        tf = getattr(item, "teacher_feedback", "") or ""
        if tf == "pending":
            tool_call = (item.metadata or {}).get("tool_call")
            name = tool_call.get("name", "") if isinstance(tool_call, dict) else ""
            if name in ("click_at", "scroll"):
                self._prompt_supervisor(item)
            else:
                self.submit_supervised_verdict(str(item.id), "pass")
        # Settle detection (top-level main turn with no further tool call). For
        # tool-call turns this is a no-op (the chain continues).
        self.handle_completed_message(item)

    def _prompt_supervisor(self, item: TextMessage) -> None:
        """Show the supervision window and arm the gate for an input-action turn.

        Non-blocking: the verdict arrives later via the supervised routes, which
        call submit_supervised_verdict.
        """
        review_text = str(item.text or "")
        show_supervised_review(review_text)

        from assistant.services.segment_parsing import extract_shapes_from_item

        preview_shapes = (
            extract_shapes_from_item(item) + self._extract_shapes_from_interpreter()
        )
        if preview_shapes:
            self._show_preview_shapes(preview_shapes)

        self.supervised_gate.arm(str(item.id))

    def submit_supervised_verdict(
        self, item_id: str, verdict: str, correction_text: str = ""
    ) -> None:
        """Write a supervision verdict onto a held turn + record a FeedbackItem.

        The teacher_feedback write syncs to the server, which resumes
        (pass/correct) or hard-suppresses (error) the withheld continuation.
        """
        ctx = self.context_manager
        item = ctx._store.find_one(id=item_id)
        if item is None:
            log.warning("submit_supervised_verdict: turn %s not found", item_id)
            return
        if getattr(item, "teacher_feedback", "") != "pending":
            # Already graded (e.g. a duplicate route call) — ignore.
            return

        updated = item.copy()
        updated.teacher_feedback = verdict
        ctx.update(updated)

        if verdict == "error":
            self._insert_feedback_and_retry(item, correction_text)
        else:
            self._insert_feedback(item, verdict)

        clear_supervised_review()
        vii.app.view_state.overlay_VS.shapes = []
        self.supervised_gate.clear()

    def _insert_feedback(self, item: TextMessage, assessment: str) -> None:
        """Insert a FeedbackItem for training data."""
        from assistant.inference.context import FeedbackItem

        feedback = FeedbackItem(
            assessment=assessment,
            source_item_id=item.id,
        )
        ctx = self.context_manager
        feedback.position = ctx.next_position()
        feedback.previous_message_id = str(item.id)
        ctx.insert(feedback)

    def _insert_feedback_and_retry(
        self, item: TextMessage, correction_text: str
    ) -> None:
        """Error verdict recovery (client-owned). The server hard-suppresses the
        bad turn's own continuation; the client mints a fresh continuation off
        the supervisor's correction. STUB: minimal recovery for now.
        """
        from assistant.inference.context import FeedbackItem

        ctx = self.context_manager

        feedback = FeedbackItem(
            assessment="error",
            correction_text=correction_text,
            source_item_id=item.id,
        )
        feedback.position = ctx.next_position()
        feedback.previous_message_id = str(item.id)
        ctx.insert(feedback)

        # Insert correction as a user message so the model sees it
        correction_item = TextMessage(
            text=f"[Supervisor correction]: {correction_text}",
        )
        correction_item.origin = "user"
        correction_item.position = ctx.next_position()
        correction_item.previous_message_id = str(feedback.id)
        correction_item.metadata = {"focus_mode": "main"}
        ctx.insert(correction_item)

        # Fresh continuation request (re-armed for review in supervised mode)
        req_item = TextMessage()
        req_item.position = ctx.next_position()
        req_item.origin = "assistant"
        req_item.previous_message_id = str(correction_item.id)
        req_item.request = {"stream": True, "focus_mode": "main"}
        req_item.metadata = {"focus_mode": "main"}
        if self._supervised():
            req_item.teacher_feedback = "pending"
        ctx.insert(req_item)

    def _insert_image_tool_item(
        self,
        image_b64: str,
        width: int,
        height: int,
        *,
        origin: str,
    ) -> ImageMessage:
        ctx = self.context_manager
        item = ImageMessage()
        item.position = ctx.next_position()
        item.image_b64 = image_b64
        item.width = width
        item.height = height
        item.size = width * height
        item.origin = origin
        item.metadata = {"visible_to": list(PERCEPTION_MODES)}
        ctx.insert(item)
        return item

    def _insert_screen_capture(self) -> None:
        """Capture the watched screen, resize, and insert as an ImageItem."""
        capture = vii.app.view_state.capture_screen_info
        screen_name = capture.name if capture else ""
        screens = QGuiApplication.screens()
        screen = None
        for s in screens:
            if s.name() == screen_name:
                screen = s
                break
        if screen is None and screens:
            screen = screens[0]
        if screen is None:
            log.warning("_insert_screen_capture: no screen available")
            return

        qpixmap = take_screenshot(screen)
        if qpixmap is None or qpixmap.isNull():
            log.warning("_insert_screen_capture: screenshot failed")
            return

        pil_img = qpixmap_to_pil(qpixmap)
        # Resize to model resolution
        target_w, target_h = get_resolution_for_model(vii.get_config().selected_model)
        pil_img, _ = resize_to_target(pil_img, target_w, target_h)

        # Encode to base64
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        self._insert_image_tool_item(
            image_b64,
            pil_img.width,
            pil_img.height,
            origin="screenshot",
        )

    def _get_last_screenshot_b64(self) -> str | None:
        """Get the base64-encoded last screenshot from context."""
        screenshot = self.context_manager.last_screenshot_item()
        if screenshot is None:
            screenshot = self.context_manager.last_image_item()
        if screenshot is None:
            return None
        return screenshot.image_b64 if screenshot.image_b64 else None

    def _crop_screenshot(
        self, screenshot_b64: str, bbox_xyxy: tuple[int, int, int, int]
    ) -> tuple[str, int, int]:
        """Crop a base64 screenshot at the given bbox. Returns (b64, width, height)."""
        raw = base64.b64decode(screenshot_b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")

        x1, y1, x2, y2 = bbox_xyxy
        # Scale from Qwen 0..1000 grid to actual image pixels
        img_w, img_h = img.size
        px1 = int(x1 * img_w / 1000)
        py1 = int(y1 * img_h / 1000)
        px2 = int(x2 * img_w / 1000)
        py2 = int(y2 * img_h / 1000)

        # Order, clamp to image bounds, and guarantee a non-empty box so PIL
        # never raises "tile cannot extend outside image" on a reversed or
        # out-of-range bbox from the model.
        left, right = sorted((px1, px2))
        top, bottom = sorted((py1, py2))
        left = max(0, min(left, img_w - 1))
        top = max(0, min(top, img_h - 1))
        right = max(left + 1, min(right, img_w))
        bottom = max(top + 1, min(bottom, img_h))

        cropped = img.crop((left, top, right, bottom))
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return b64, cropped.width, cropped.height

    _MAX_SCROLL_STEPS = 10

    async def _exec_scroll(
        self, coordinate: list[int], direction: str, amount: int
    ) -> str:
        """Scroll at explicit coordinate (0-1000 grid) via ydotool."""
        from assistant.services import input_control

        if amount == 0:
            return "scroll: 0 amount, nothing to do"

        # Clamp to prevent runaway scrolling
        if amount > self._MAX_SCROLL_STEPS:
            log.warning("scroll: clamped %d to %d", amount, self._MAX_SCROLL_STEPS)
            amount = self._MAX_SCROLL_STEPS

        # Convert direction + amount to signed steps (positive=up, negative=down)
        steps = amount if direction == "up" else -amount

        # Convert 0-1000 grid to screen pixels (with offset)
        x, y = coordinate[0], coordinate[1]
        ox, oy, screen_w, screen_h = self._resolve_screen_geometry()
        px = ox + int(x / 1000 * screen_w)
        py = oy + int(y / 1000 * screen_h)

        # Move pointer to position then scroll
        if not await input_control.move_pointer(px, py):
            return f"error: scroll failed: could not move pointer to ({px}, {py})"
        if not await input_control.scroll(steps):
            return f"error: scroll failed at ({px}, {py})"

        self._pointer_x = px
        self._pointer_y = py
        return f"scrolled {direction} {amount} clicks at ({x}, {y})"
