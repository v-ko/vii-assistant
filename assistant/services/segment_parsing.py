"""Pure parsing utilities and agent tool definitions for segment/overlay output.

This module holds the HybridFunctionInterpreter instance, tool function
registrations, and all stateless parsing/extraction helpers.  It has no
dependency on the facade singleton or PySide6, so it can be imported freely
without circular-dep issues.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from PIL import Image
from sivkit import get_logger
from sivkit.util.rectangle import Rectangle

from assistant.inference.context import TextItem
from assistant.inference.function_interpreter import HybridFunctionInterpreter

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# HFI instance + registered tool stubs
# ---------------------------------------------------------------------------

hfi = HybridFunctionInterpreter()


@hfi.function(description="Locate region using xyxy format (x1, y1, x2, y2)")
def locate(x1: int, y1: int, x2: int, y2: int) -> Rectangle:
    """locate function expecting xyxy format (x1, y1, x2, y2) like models output.

    Converts to xywh format for Rectangle.
    """
    width = x2 - x1
    height = y2 - y1
    return Rectangle(x1, y1, width, height)


@hfi.function(
    description="Create rectangle using xywh format (x, y, width, height)",
)
def bbox(x: int, y: int, width: int, height: int) -> Rectangle:
    return Rectangle(x, y, width, height)


@hfi.function(description="Mark a point at (x, y) in Qwen grid coordinates")
def point(x: int, y: int) -> tuple[int, int]:
    return (x, y)


# --- Agent tool function stubs ---
# Registered on hfi so execute_block can dispatch them.
# Actual execution is handled by HybridSegmentService methods.


@hfi.function(
    name="crop_image",
    description="Crop an image from the current screenshot at the given bbox (0-1000 grid). Returns base64 image.",
)
def crop_image(bbox: list) -> str:
    return str(bbox)


@hfi.function(
    name="curator_push",
    description="Push an item (dict) to a named curator feed for review.",
)
def curator_push(feed: str, item: Any) -> str:
    return f"pushed to {feed}"


AGENT_TOOLS = frozenset(
    {
        "crop_image",
        "curator_push",
    }
)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class SegmentOutput(TypedDict, total=False):
    bbox: list[Rectangle]
    image: list[Rectangle] | list[Image.Image]
    points: list[tuple[int, int]]


@dataclass(frozen=True)
class ExtractedToolCall:
    id: str
    message_id: str
    ordinal: int
    name: str
    arg_exprs: list[str]
    assign_to: str | None = None
    source_line: str = ""
    line_number: int = 0


@dataclass
class ToolCallResult:
    call_id: str
    name: str
    status: Literal["success", "error"]
    value: Any = None
    message: str = ""
    context_item_id: str | None = None


# ---------------------------------------------------------------------------
# Segment output parsing (overlays)
# ---------------------------------------------------------------------------


def parse_segment_output(text: str, bbox_format: str = "xyxy") -> SegmentOutput:
    out: SegmentOutput = {"bbox": [], "image": [], "points": []}

    _parse_json_blocks(text, out, bbox_format)

    for line in text.splitlines():
        if not line.strip():
            continue

        parsed = hfi.parse_line(line)
        if parsed is None:
            continue

        lhs, func_name, args = parsed
        log.debug(f"Parsing segment line: lhs={lhs}, func={func_name}, args={args}")

        # Skip agent tool calls — they are handled separately
        if func_name in AGENT_TOOLS:
            continue

        try:
            result = hfi.execute_call(func_name, args)
        except Exception as e:
            log.error(
                f"Segment parse error: {e}"
                + (f" lhs={lhs}" if lhs else "")
                + f" func={func_name} args={args}"
            )
            continue

        if func_name == "point":
            log.info(f"Parsed point({', '.join(args)}) -> {result}")
            out["points"].append(result)  # type: ignore[union-attr]
            continue

        if not isinstance(result, Rectangle):
            log.error(
                f"Function {func_name} did not return Rectangle, got {type(result)}"
            )
            continue

        if func_name == "bbox":
            log.info(
                f"Parsed bbox({', '.join(args)}) [xywh] ->"
                f" Rectangle{result.as_tuple()}"
            )
        elif func_name == "locate":
            log.info(
                f"Parsed locate({', '.join(args)}) [xyxy] ->"
                f" Rectangle{result.as_tuple()} [xywh]"
            )

        if lhs:
            if lhs not in ("output.bbox", "output.image"):
                log.error(f"Segment parse ignore (invalid lhs root): {lhs}")
                continue
            if lhs.endswith("bbox"):
                out["bbox"].append(result)  # type: ignore[union-attr]
            else:
                out["image"].append(result)  # type: ignore[union-attr]
        else:
            if func_name == "bbox":
                out["bbox"].append(result)  # type: ignore[union-attr]
            elif func_name == "locate":
                out["image"].append(result)  # type: ignore[union-attr]

    return out


def _parse_json_blocks(text: str, out: SegmentOutput, bbox_format: str) -> None:
    # 1) Fenced ```json ... ``` blocks
    for m in re.finditer(r"```json\s*\n(.*?)```", text, re.DOTALL):
        raw = m.group(1).strip()
        _try_load_json_block(raw, out, bbox_format)

    # 2) Bare JSON objects/arrays not inside code fences
    stripped = re.sub(r"```json\s*\n.*?```", "", text, flags=re.DOTALL)
    for m in re.finditer(
        r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])",
        stripped,
        re.DOTALL,
    ):
        raw = m.group(1).strip()
        _try_load_json_block(raw, out, bbox_format)


def _try_load_json_block(raw: str, out: SegmentOutput, bbox_format: str) -> None:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return

    items: list[dict] = []
    if isinstance(data, dict):
        items = [data]
    elif isinstance(data, list):
        items = [d for d in data if isinstance(d, dict)]
    else:
        log.error(f"JSON block has unexpected type: {type(data)}")
        return

    for item in items:
        _parse_json_item(item, out, bbox_format)


def extract_bbox_from_dict(
    item: dict, bbox_format: str = "xyxy"
) -> tuple[int, int, int, int] | None:
    """Extract an xyxy bbox tuple from a JSON dict.

    Recognises keys: bbox, bbox_2d (xyxy), box_2d (yxyx for Gemma).
    Returns (x1, y1, x2, y2) or None.
    """
    bbox_val = item.get("bbox") or item.get("bbox_2d")
    is_yxyx = False
    if bbox_val is None:
        bbox_val = item.get("box_2d")
        if bbox_val is not None:
            is_yxyx = True  # Gemma box_2d is always yxyx regardless of config
    if not isinstance(bbox_val, list) or len(bbox_val) != 4:
        return None
    try:
        coords = [int(v) for v in bbox_val]
    except (ValueError, TypeError):
        return None
    # If from box_2d key or model config says yxyx, swap to xyxy
    if is_yxyx or bbox_format == "yxyx":
        return (coords[1], coords[0], coords[3], coords[2])
    return (coords[0], coords[1], coords[2], coords[3])


def _parse_json_item(item: dict, out: SegmentOutput, bbox_format: str) -> None:
    xyxy = extract_bbox_from_dict(item, bbox_format)
    if xyxy is not None:
        x1, y1, x2, y2 = xyxy
        rect = Rectangle(x1, y1, x2 - x1, y2 - y1)
        log.info(f"JSON bbox -> xyxy={xyxy} -> Rectangle{rect.as_tuple()}")
        out["bbox"].append(rect)  # type: ignore[union-attr]

    point_val = item.get("point_2d") or item.get("point")
    if isinstance(point_val, list) and len(point_val) == 2:
        try:
            px, py = int(point_val[0]), int(point_val[1])
            log.info(f"JSON point [{px}, {py}]")
            out["points"].append((px, py))  # type: ignore[union-attr]
        except (ValueError, TypeError) as e:
            log.error(f"JSON point conversion error: {e}")


# ---------------------------------------------------------------------------
# Tool call extraction / ID utilities
# ---------------------------------------------------------------------------


def extract_tool_calls_from_message(item: TextItem) -> list[ExtractedToolCall]:
    calls: list[ExtractedToolCall] = []
    message_id = str(item.id)

    for line_number, raw_line in enumerate(str(item.text or "").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            tree = ast.parse(line, mode="exec")
        except SyntaxError:
            continue

        for stmt in tree.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                value = stmt.value
                if isinstance(value, ast.Call):
                    call = _tool_call_from_ast(
                        message_id=message_id,
                        ordinal=len(calls),
                        call_node=value,
                        assign_to=hfi._target_str(target),
                        source_line=line,
                        line_number=line_number,
                    )
                    if call is not None:
                        calls.append(call)
                        continue

                resolved = hfi._resolve_node(value)
                hfi._assign_to_target(target, resolved)

            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call = _tool_call_from_ast(
                    message_id=message_id,
                    ordinal=len(calls),
                    call_node=stmt.value,
                    assign_to=None,
                    source_line=line,
                    line_number=line_number,
                )
                if call is not None:
                    calls.append(call)

    return calls


def _tool_call_from_ast(
    *,
    message_id: str,
    ordinal: int,
    call_node: ast.Call,
    assign_to: str | None,
    source_line: str,
    line_number: int,
) -> ExtractedToolCall | None:
    name = hfi._func_name_from_node(call_node.func)
    if name not in AGENT_TOOLS:
        return None

    arg_exprs = [ast.unparse(arg) for arg in call_node.args]
    call_id = tool_call_id(
        message_id=message_id,
        ordinal=ordinal,
        name=name,
        arg_exprs=arg_exprs,
        assign_to=assign_to,
    )
    return ExtractedToolCall(
        id=call_id,
        message_id=message_id,
        ordinal=ordinal,
        name=name,
        arg_exprs=arg_exprs,
        assign_to=assign_to,
        source_line=source_line,
        line_number=line_number,
    )


def message_tool_run_id(item: TextItem) -> str:
    payload = json.dumps(
        {
            "message_id": str(item.id),
            "text": item.text,
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tool_call_id(
    *,
    message_id: str,
    ordinal: int,
    name: str,
    arg_exprs: list[str],
    assign_to: str | None,
) -> str:
    payload = json.dumps(
        {
            "message_id": message_id,
            "ordinal": ordinal,
            "name": name,
            "assign_to": assign_to or "",
            "args": [_normalize_expr(expr) for expr in arg_exprs],
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_expr(expr: str) -> str:
    try:
        return ast.unparse(ast.parse(expr, mode="eval").body)
    except SyntaxError:
        return " ".join(expr.split())


def format_tool_result_message(
    source_item: TextItem, results: list[ToolCallResult]
) -> str:
    lines = [f"[tool results for assistant message {source_item.id}]"]
    for result in results:
        short_id = result.call_id[:12]
        lines.append("")
        lines.append(f"{result.name}#{short_id}: {result.status}")
        if result.context_item_id:
            lines.append(f"item_id: {result.context_item_id}")
        if result.message:
            lines.append(f"message: {result.message}")
    return "\n".join(lines)
