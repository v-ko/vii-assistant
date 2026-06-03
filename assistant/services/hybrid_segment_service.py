from __future__ import annotations

import asyncio
import base64
import io
import json
import re
from typing import Any

from PIL import Image
from PySide6.QtGui import QGuiApplication
from sivkit import get_logger
from sivkit.libs.model import dump_to_dict
from sivkit.storage.change import Change
from sivkit.storage.delta import Delta
from sivkit.util.rectangle import Rectangle

from assistant.facade import vii
from assistant.image_ops import resize_to_target, scale_qwen_bbox_xyxy, scale_qwen_point
from assistant.inference.context import (
    ContextItem,
    ImageItem,
    TextItem,
)
from assistant.inference.context_store import OP_SEP
from assistant.model_configs import get_resolution_for_model
from assistant.overlay_actions import (
    clear_pending_actions,
    show_pending_actions,
)
from assistant.services.action_gate import ActionGate, PendingAction
from assistant.services.curator_client import push_item
from assistant.services.overlay_manager import OverlayManager
from assistant.services.segment_parsing import (
    ExtractedToolCall,
    ToolCallResult,
    extract_tool_calls_from_message,
    format_tool_result_message,
    hfi,
    message_tool_run_id,
    parse_segment_output,
)
from assistant.util import Shape, get_screen_by_name
from assistant.utils.capture_utils import take_screenshot
from assistant.utils.image_utils import qpixmap_to_pil
from assistant.view_states.settings import ExecutionMode

log = get_logger(__name__)


class HybridSegmentService:
    """Watches context changes for overlay updates and agent tool execution.

    Pure parsing/extraction lives in segment_parsing module.
    This class manages in-flight state and performs side effects
    (overlay updates, tool invocation, context insertion).
    """

    MAX_TURNS = 25

    def __init__(self) -> None:
        self._processed_tool_call_ids: set[str] = set()
        self._in_flight_tool_call_ids: set[str] = set()
        self._processed_message_tool_runs: set[str] = set()
        self._in_flight_message_tool_runs: set[str] = set()
        self._turn_count: int = 0
        self._stopped: bool = False
        # Non-reentrant guard — only one processing chain at a time
        self._processing: bool = False
        # Last known pointer position (screen coords) for scroll
        self._pointer_x: int | None = None
        self._pointer_y: int | None = None
        # Action gate for USER_APPROVE mode
        self.action_gate = ActionGate()
        # Guard overlays for AUTO mode
        self.overlay_manager = OverlayManager()

    @property
    def context_manager(self):
        return vii.project_manager.context_manager

    def reset_turns(self) -> None:
        self._turn_count = 0
        self._stopped = False

    def changed_context_items(self, delta: Delta) -> list[ContextItem]:
        items: list[ContextItem] = []
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
            if isinstance(entity, ContextItem):
                items.append(entity)
        return items

    # ------------------------------------------------------------------
    # Focus-mode client-side execution (tool call dispatch)
    # ------------------------------------------------------------------

    async def process_client_execution_request(self, item: TextItem) -> None:
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

        log.info(
            "Client execution request: item=%s mode=%s caller=%s",
            item_id,
            focus_mode,
            caller_mode,
        )

        # Gate action tools in user-approve mode
        is_action = focus_mode in ("click_at", "scroll")
        if is_action:
            settings = vii.app.view_state.settings_VS
            if settings.execution_mode == ExecutionMode.USER_APPROVE.value:
                desc = f"{focus_mode}({arguments})"
                pending = [
                    PendingAction(
                        call_id=item_id, tool_name=focus_mode, description=desc
                    )
                ]
                show_pending_actions(
                    [f"{a.tool_name}({a.description})" for a in pending]
                )
                gate_result = await self.action_gate.await_confirmation(pending)
                clear_pending_actions()
                if not gate_result.confirmed:
                    result_text = "Action cancelled by user"
                    self._insert_result_and_continue(
                        result_text, item_id, caller_mode, is_action=False
                    )
                    return

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

        # Insert result visible to caller mode
        result_item = TextItem()
        result_item.position = ctx.next_position()
        result_item.origin = "tool"
        result_item.text = result_text
        result_item.metadata = {
            "visible_to": [caller_mode],
            "source_item_id": source_item_id,
        }
        ctx.insert(result_item)

        # Insert a screenshot after navigation actions
        if is_action:
            self._insert_screen_capture()

        # Check loop limit
        self._turn_count += 1
        if self._turn_count >= self.MAX_TURNS:
            log.warning(
                "max_turns=%d reached in client execution, stopping", self.MAX_TURNS
            )
            vii.app.view_state.settings_VS.assistant_working = False
            return

        # Continue the caller mode
        req_item = TextItem()
        req_item.position = ctx.next_position()
        req_item.origin = "assistant"
        req_item.request = {
            "stream": True,
            "focus_mode": caller_mode,
        }
        req_item.metadata = {"focus_mode": caller_mode}
        ctx.insert(req_item)

        log.info(
            "Client execution complete, continuing '%s' (turn %d/%d)",
            caller_mode,
            self._turn_count,
            self.MAX_TURNS,
        )

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

            return "ok"
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
            log.info(f"Converted to {len(shapes)} overlay shapes")
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
        log.info(f"Screen geometry: {x},{y} {w}x{h}")
        return x, y, w, h

    def _resolve_input_resolution(
        self, fallback_w: int, fallback_h: int
    ) -> tuple[int, int]:
        """Get the resolution the model saw — from the last image item or model config."""
        for it in self.context_manager.items_reversed():
            if isinstance(it, ImageItem):
                log.info(f"Found image size: {it.width}x{it.height}")
                return it.width, it.height

        log.info("No image metadata found, using model's configured resolution")
        model_key = vii.get_config().selected_model
        w, h = get_resolution_for_model(model_key)
        log.info(f"Model resolution: {w}x{h}")
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

        for r in rects:
            x1 = r.x()
            y1 = r.y()
            x2 = r.right()
            y2 = r.bottom()
            log.info(
                f"Processing rectangle: x1={x1}, y1={y1}, x2={x2}, y2={y2} (from Qwen"
                " grid)"
            )
            sx, sy, ex, ey = scale_qwen_bbox_xyxy(
                (int(x1), int(y1), int(x2), int(y2)),
                input_w=input_w,
                input_h=input_h,
                orig_w=output_w,
                orig_h=output_h,
                coords_are_qwen_grid=True,
            )
            log.info(f"Scaled to screen: sx={sx}, sy={sy}, ex={ex}, ey={ey}")
            w = max(0, ex - sx)
            h = max(0, ey - sy)
            final = Rectangle(sx, sy, w, h)
            gx, gy, gw, gh = final.as_tuple()
            log.info(f"Final shape geometry: x={gx}, y={gy}, w={gw}, h={gh}")
            rect_shape: Shape = {
                "type": "rect",
                "geometry": (int(gx), int(gy), int(gw), int(gh)),
                "color": "#FF0000",
            }
            shapes.append(rect_shape)

        for pt in points or []:
            sx, sy = scale_qwen_point(
                pt,
                input_w=input_w,
                input_h=input_h,
                orig_w=output_w,
                orig_h=output_h,
                coords_are_qwen_grid=True,
            )
            log.info(f"Scaled point {pt} to screen: ({sx}, {sy})")
            point_shape: Shape = {
                "type": "point",
                "geometry": (int(sx), int(sy)),
                "color": "#00FF00",
            }
            shapes.append(point_shape)

        return shapes

    # ------------------------------------------------------------------
    # Agent tool execution
    # ------------------------------------------------------------------

    async def process_completed_assistant_message(self, item: TextItem) -> None:
        """Process a completed assistant message. Non-reentrant: if already
        processing, the call is skipped."""
        if self._processing:
            return
        self._processing = True
        try:
            await self._process_completed_assistant_message(item)
        finally:
            self._processing = False

    async def _process_completed_assistant_message(self, item: TextItem) -> None:
        text = str(item.text or "")
        if not text.strip():
            return

        # Don't process cancelled messages
        if isinstance(item.request, dict) and item.request.get("cancelled_by_user"):
            return

        run_id = message_tool_run_id(item)
        if run_id in self._processed_message_tool_runs:
            return
        if run_id in self._in_flight_message_tool_runs:
            return

        calls = extract_tool_calls_from_message(item)
        if not calls:
            # Final response — no more work to do
            vii.app.view_state.settings_VS.assistant_working = False
            return

        pending_calls = [
            call
            for call in calls
            if call.id not in self._processed_tool_call_ids
            and call.id not in self._in_flight_tool_call_ids
        ]
        if not pending_calls:
            self._processed_message_tool_runs.add(run_id)
            return

        self._in_flight_message_tool_runs.add(run_id)
        for call in pending_calls:
            self._in_flight_tool_call_ids.add(call.id)

        try:
            results = await self._invoke_tool_calls_for_message(item, pending_calls)
            if results:
                self._insert_tool_results_and_maybe_continue(item, results)
                self._processed_message_tool_runs.add(run_id)
                for result in results:
                    self._processed_tool_call_ids.add(result.call_id)
        finally:
            self._in_flight_message_tool_runs.discard(run_id)
            for call in pending_calls:
                self._in_flight_tool_call_ids.discard(call.id)

    # Tools that perform side-effects (input actions) and need approval
    _ACTION_TOOLS = {"click_at", "scroll", "move_pointer", "type_text", "key_combo"}

    async def _invoke_tool_calls_for_message(
        self, item: TextItem, calls: list[ExtractedToolCall]
    ) -> list[ToolCallResult]:
        results: list[ToolCallResult] = []

        # Separate action tools from read-only tools
        action_calls = [c for c in calls if c.name in self._ACTION_TOOLS]
        read_only_calls = [c for c in calls if c.name not in self._ACTION_TOOLS]

        # Execute read-only tools immediately
        for call in read_only_calls:
            if self._stopped:
                break
            try:
                resolved_args = [hfi.resolve_arg(expr) for expr in call.arg_exprs]
                result = await self._invoke_tool_call(call, resolved_args)
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Tool call failed: message=%s call=%s tool=%s error=%s",
                    item.id,
                    call.id,
                    call.name,
                    exc,
                    exc_info=True,
                )
                result = ToolCallResult(
                    call_id=call.id,
                    name=call.name,
                    status="error",
                    message=str(exc),
                )
            results.append(result)

        # Gate action tools based on execution mode
        if action_calls and not self._stopped:
            settings = vii.app.view_state.settings_VS
            mode = settings.execution_mode

            if mode == ExecutionMode.USER_APPROVE.value:
                # Show pending actions in overlay and wait for user confirmation
                pending = [
                    PendingAction(
                        call_id=c.id,
                        tool_name=c.name,
                        description=str(c.arg_exprs),
                    )
                    for c in action_calls
                ]
                show_pending_actions(
                    [f"{a.tool_name}({a.description})" for a in pending]
                )
                gate_result = await self.action_gate.await_confirmation(pending)
                clear_pending_actions()

                if not gate_result.confirmed:
                    # User rejected — mark all as cancelled
                    for call in action_calls:
                        results.append(
                            ToolCallResult(
                                call_id=call.id,
                                name=call.name,
                                status="error",
                                message="cancelled by user",
                            )
                        )
                    return results

            # Execute action tools (auto mode or after user confirmed)
            for call in action_calls:
                if self._stopped:
                    break
                try:
                    resolved_args = [hfi.resolve_arg(expr) for expr in call.arg_exprs]
                    result = await self._invoke_tool_call(call, resolved_args)
                except Exception as exc:  # noqa: BLE001
                    log.error(
                        "Tool call failed: message=%s call=%s tool=%s error=%s",
                        item.id,
                        call.id,
                        call.name,
                        exc,
                        exc_info=True,
                    )
                    result = ToolCallResult(
                        call_id=call.id,
                        name=call.name,
                        status="error",
                        message=str(exc),
                    )
                results.append(result)

        return results

    async def _invoke_tool_call(
        self, call: ExtractedToolCall, resolved_args: list[Any]
    ) -> ToolCallResult:
        if call.name == "crop_image":
            bbox = resolved_args[0] if resolved_args else []
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                return ToolCallResult(
                    call_id=call.id,
                    name=call.name,
                    status="error",
                    message="crop_image requires [x1, y1, x2, y2] bbox",
                )
            screenshot_b64 = self._get_last_screenshot_b64()
            if screenshot_b64 is None:
                return ToolCallResult(
                    call_id=call.id,
                    name=call.name,
                    status="error",
                    message="crop_image: no screenshot available",
                )
            bbox_tuple = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            cropped_b64, width, height = self._crop_screenshot(
                screenshot_b64, bbox_tuple
            )
            self._update_variable(call.assign_to, cropped_b64)
            self._update_variable(call.assign_to, cropped_b64)
            image_item = self._insert_image_tool_item(
                cropped_b64, width, height, origin="tool"
            )
            return ToolCallResult(
                call_id=call.id,
                name=call.name,
                status="success",
                value=cropped_b64,
                context_item_id=str(image_item.id),
                message=f"image item {image_item.id} ({width}x{height})",
            )

        if call.name == "curator_push":
            feed_name = str(resolved_args[0]) if resolved_args else "default"
            content = resolved_args[1] if len(resolved_args) > 1 else {}
            metadata = resolved_args[2] if len(resolved_args) > 2 else None
            if not isinstance(content, dict):
                content = {}
            if metadata is not None and not isinstance(metadata, dict):
                metadata = None
            # Coerce content values to strings (LLM may produce non-string values)
            for key in ("text", "title", "url", "image", "video"):
                if key in content and content[key] is not None:
                    content[key] = str(content[key])
            # Ensure there's a text field so the item is renderable
            if "text" not in content and "url" not in content:
                content["text"] = content.get("title", "(no content)")
            response = await push_item(feed_name, content, metadata)
            if response.get("status") != "ok":
                return ToolCallResult(
                    call_id=call.id,
                    name=call.name,
                    status="error",
                    message=str(response.get("error_message") or "curator push failed"),
                )
            return ToolCallResult(
                call_id=call.id,
                name=call.name,
                status="success",
                message=f"item pushed to feed '{feed_name}'",
            )

        if call.name == "move_pointer":
            description = str(resolved_args[0]) if resolved_args else ""
            result_msg = await self._exec_move_pointer(description)
            if result_msg is None:
                return ToolCallResult(
                    call_id=call.id,
                    name=call.name,
                    status="error",
                    message="move_pointer: failed to resolve position",
                )
            return ToolCallResult(
                call_id=call.id,
                name=call.name,
                status="success",
                message=result_msg,
            )

        if call.name == "click_at":
            description = str(resolved_args[0]) if resolved_args else ""
            result_msg = await self._exec_click_at(description)
            if result_msg is None:
                return ToolCallResult(
                    call_id=call.id,
                    name=call.name,
                    status="error",
                    message="click_at: failed to resolve position",
                )
            return ToolCallResult(
                call_id=call.id,
                name=call.name,
                status="success",
                message=result_msg,
            )

        if call.name == "scroll":
            coordinate = list(resolved_args[0]) if resolved_args else [500, 500]
            direction = str(resolved_args[1]) if len(resolved_args) > 1 else "down"
            amount = int(resolved_args[2]) if len(resolved_args) > 2 else 3
            result_msg = await self._exec_scroll(coordinate, direction, amount)
            return ToolCallResult(
                call_id=call.id,
                name=call.name,
                status="success" if "error" not in result_msg.lower() else "error",
                message=result_msg,
            )

        return ToolCallResult(
            call_id=call.id,
            name=call.name,
            status="error",
            message=f"unknown tool: {call.name}",
        )

    def _insert_image_tool_item(
        self, image_b64: str, width: int, height: int, *, origin: str
    ) -> ImageItem:
        ctx = self.context_manager
        item = ImageItem()
        item.position = ctx.next_position()
        item.image_b64 = image_b64
        item.width = width
        item.height = height
        item.size = width * height
        item.origin = origin
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
            image_b64, pil_img.width, pil_img.height, origin="screenshot"
        )
        log.info(
            "Inserted screen capture (%dx%d) after tool execution",
            pil_img.width,
            pil_img.height,
        )

    def _insert_tool_results_and_maybe_continue(
        self, source_item: TextItem, results: list[ToolCallResult]
    ) -> None:
        if not results:
            return

        ctx = self.context_manager
        result_item = TextItem()
        result_item.position = ctx.next_position()
        result_item.origin = "tool"
        result_item.text = format_tool_result_message(source_item, results)
        result_item.metadata = {
            "source_assistant_item_id": str(source_item.id),
            "tool_call_ids": [result.call_id for result in results],
        }
        ctx.insert(result_item)

        # Insert a screenshot of the current screen state after tool execution
        self._insert_screen_capture()

        # Check if stopped
        if self._stopped:
            log.info("Stopped — not continuing after tool results")
            vii.app.view_state.settings_VS.assistant_working = False
            return

        # Auto-continuation: trigger next generation after tool execution
        self._turn_count += 1
        if self._turn_count >= self.MAX_TURNS:
            log.warning(
                "max_turns=%d reached, stopping auto-continuation", self.MAX_TURNS
            )
            vii.app.view_state.settings_VS.assistant_working = False
            return

        req_item = TextItem()
        req_item.position = ctx.next_position()
        req_item.origin = "assistant"
        req_item.request = {"stream": True}
        ctx.insert(req_item)
        log.info(
            "Triggered continuation generation (turn %d/%d)",
            self._turn_count,
            self.MAX_TURNS,
        )

    def _update_variable(self, lhs: str | None, value: Any) -> None:
        """Update the interpreter variable for a given lhs expression."""
        if not lhs:
            return
        # Dict key assignment: var['key']
        m = re.match(r"^([A-Za-z_]\w*)\['(.+?)'\]$", lhs)
        if m:
            var_name, key = m.group(1), m.group(2)
            if var_name in hfi.variables and isinstance(hfi.variables[var_name], dict):
                hfi.variables[var_name][key] = value
        else:
            hfi.variables[lhs] = value

    async def _resolve_bbox_via_subagent(
        self, description: str, screenshot_b64: str
    ) -> tuple[int, int, int, int] | None:
        """Call /infer to resolve a semantic description to a bounding box.

        Returns (x1, y1, x2, y2) in pixel coordinates of the screenshot, or None.
        """
        # Build a small context with the screenshot + prompt
        img_item = ImageItem()
        img_item.position = 0
        img_item.image_b64 = screenshot_b64
        img_item.origin = "screenshot"
        prompt = (
            f"Locate {description}, " f"report the bbox coordinates in JSON format."
        )
        text_item = TextItem()
        text_item.position = 100
        text_item.text = prompt
        text_item.origin = "user"
        # Request item to trigger generation
        req_item = TextItem()
        req_item.position = 200
        req_item.origin = "assistant"
        req_item.request = {"stream": False}

        context_data = [
            dump_to_dict(img_item),
            dump_to_dict(text_item),
            dump_to_dict(req_item),
        ]

        try:
            data = await vii.inference_client.infer(
                context_data,
                {"max_new_tokens": 64, "do_sample": False},
                timeout=30.0,
            )
        except Exception as exc:
            log.error("Sub-agent /infer call failed: %s", exc)
            return None

        if data.get("status") != "success":
            log.error("Sub-agent returned error: %s", data.get("error_message"))
            return None

        # Parse bbox from JSON response (e.g. {"bbox": [x1, y1, x2, y2]}
        # or [[x1, y1, x2, y2]] or bare [x1, y1, x2, y2])
        response_text = data.get("text", "")
        return self._parse_bbox_from_response(response_text)

    def _parse_bbox_from_response(self, text: str) -> tuple[int, int, int, int] | None:
        """Extract xyxy bbox from model response text.

        Handles formats like:
          {"bbox": [x1, y1, x2, y2]}
          [[x1, y1, x2, y2]]
          [x1, y1, x2, y2]
          ```json\n{"bbox": [...]}\n```
        """
        # Strip markdown fences if present
        stripped = re.sub(r"```json\s*\n?", "", text)
        stripped = re.sub(r"```", "", stripped).strip()

        # Try direct JSON parse
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            # Try to find a JSON-like array in the text
            m = re.search(
                r"\[\s*\[?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]?\s*\]", text
            )
            if m:
                return (
                    int(m.group(1)),
                    int(m.group(2)),
                    int(m.group(3)),
                    int(m.group(4)),
                )
            log.error("Could not parse bbox from sub-agent response: %s", text)
            return None

        # Navigate into the parsed structure
        coords: list | None = None
        if isinstance(data, dict):
            # {"bbox": [...]} or {"bbox_2d": [...]}
            coords = data.get("bbox") or data.get("bbox_2d")
        elif isinstance(data, list):
            if data and isinstance(data[0], dict):
                # [{"bbox_2d": [x1, y1, x2, y2], "label": ...}]
                coords = data[0].get("bbox") or data[0].get("bbox_2d")
            elif data and isinstance(data[0], list):
                # [[x1, y1, x2, y2]]
                coords = data[0]
            elif len(data) == 4:
                # [x1, y1, x2, y2]
                coords = data

        if coords and len(coords) == 4:
            try:
                return (int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))
            except (ValueError, TypeError) as e:
                log.error("Bad bbox values: %s (%s)", coords, e)
                return None

        log.error("Could not extract bbox from parsed response: %s", text)
        return None

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

        # Clamp
        px1 = max(0, min(px1, img_w))
        py1 = max(0, min(py1, img_h))
        px2 = max(0, min(px2, img_w))
        py2 = max(0, min(py2, img_h))

        cropped = img.crop((px1, py1, px2, py2))
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return b64, cropped.width, cropped.height

    # --- MPX pointer tools ---

    # --- MPX pointer tools ---

    def _resolve_bbox_center_to_screen(
        self, bbox_xyxy: tuple[int, int, int, int]
    ) -> tuple[int, int]:
        """Convert Qwen grid bbox center to screen pixel coordinates."""
        x1, y1, x2, y2 = bbox_xyxy
        center_qwen_x = (x1 + x2) // 2
        center_qwen_y = (y1 + y2) // 2
        ox, oy, screen_w, screen_h = self._resolve_screen_geometry()
        # Direct Qwen grid (0-1000) → screen coordinate mapping
        screen_x = ox + int(center_qwen_x / 1000 * screen_w)
        screen_y = oy + int(center_qwen_y / 1000 * screen_h)
        return screen_x, screen_y

    async def _exec_move_pointer(self, description: str) -> str | None:
        """Move the assistant MPX pointer to the described element."""
        screenshot_b64 = self._get_last_screenshot_b64()
        if screenshot_b64 is None:
            log.error("move_pointer: no screenshot available")
            return None

        bbox_coords = await self._resolve_bbox_via_subagent(description, screenshot_b64)
        if bbox_coords is None:
            log.error("move_pointer: could not resolve '%s'", description)
            return None

        screen_x, screen_y = self._resolve_bbox_center_to_screen(bbox_coords)

        from assistant.services import input_control

        if not await input_control.move_pointer(screen_x, screen_y):
            log.error("move_pointer: ydotool move failed")
            return None

        self._pointer_x = screen_x
        self._pointer_y = screen_y
        return f"moved pointer to ({screen_x}, {screen_y})"

    async def _exec_click_at(self, description: str) -> str | None:
        """Click the assistant MPX pointer at the described element."""
        screenshot_b64 = self._get_last_screenshot_b64()
        if screenshot_b64 is None:
            log.error("click_at: no screenshot available")
            return None

        bbox_coords = await self._resolve_bbox_via_subagent(description, screenshot_b64)
        if bbox_coords is None:
            log.error("click_at: could not resolve '%s'", description)
            return None

        screen_x, screen_y = self._resolve_bbox_center_to_screen(bbox_coords)

        from assistant.services import input_control

        if not await input_control.click(screen_x, screen_y):
            log.error("click_at: ydotool click failed")
            return None

        self._pointer_x = screen_x
        self._pointer_y = screen_y
        return f"clicked at ({screen_x}, {screen_y})"

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
