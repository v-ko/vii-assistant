"""Pure parsing utilities and agent tool definitions for segment/overlay output.

This module holds the HybridFunctionInterpreter instance, tool function
registrations, and all stateless parsing/extraction helpers.  It has no
dependency on the facade singleton or PySide6, so it can be imported freely
without circular-dep issues.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any, TypedDict

from PIL import Image
from sivkit.util.rectangle import Rectangle

from assistant.inference.context import TextMessage
from assistant.inference.function_interpreter import HybridFunctionInterpreter

log = logging.getLogger(__name__)

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
# Shape extraction from agent code (AST-based)
# ---------------------------------------------------------------------------

from assistant.util import Shape


def _is_grid_coord(value: int) -> bool:
    """Check if a value looks like a Qwen 0-1000 grid coordinate."""
    return 0 <= value <= 1000


def _try_eval_int(node: ast.expr) -> int | None:
    """Try to statically evaluate an AST node to an integer."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _try_eval_int(node.operand)
        if inner is not None:
            return -inner
    return None


def _try_extract_int_list(node: ast.expr) -> list[int] | None:
    """Try to extract a list of integer constants from an AST List node."""
    if not isinstance(node, ast.List):
        return None
    values = []
    for elt in node.elts:
        v = _try_eval_int(elt)
        if v is None:
            return None
        values.append(v)
    return values


def _is_bbox_list(values: list[int]) -> bool:
    """Check if a 4-element list looks like a bbox in 0-1000 grid."""
    if len(values) != 4:
        return False
    return all(_is_grid_coord(v) for v in values)


def _is_point_list(values: list[int]) -> bool:
    """Check if a 2-element list looks like a point in 0-1000 grid."""
    if len(values) != 2:
        return False
    return all(_is_grid_coord(v) for v in values)


def extract_shapes_from_code(code: str) -> list[Shape]:
    """Extract bbox and point shapes from agent python code using AST.

    Finds:
    - crop_image([x1,y1,x2,y2]) calls → rect (xyxy)
    - locate(x1,y1,x2,y2) calls → rect (xyxy)
    - bbox(x,y,w,h) calls → rect (xywh)
    - click_at(x, y) / point(x,y) calls → point
    - 4-element list literals (0-1000 range) in assignments or args → rect (xyxy)
    - 2-element list literals in scroll coordinate arg → point

    All coordinates are in Qwen 0-1000 grid (not scaled to screen).
    """
    shapes: list[Shape] = []
    seen_lists: set[tuple] = set()  # avoid duplicates

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        return shapes

    def _add_rect_xyxy(x1: int, y1: int, x2: int, y2: int) -> None:
        key = ("rect", x1, y1, x2, y2)
        if key in seen_lists:
            return
        seen_lists.add(key)
        w = x2 - x1
        h = y2 - y1
        if w > 0 and h > 0:
            shapes.append({"type": "rect", "geometry": (x1, y1, w, h)})

    def _add_rect_xywh(x: int, y: int, w: int, h: int) -> None:
        key = ("rect", x, y, x + w, y + h)
        if key in seen_lists:
            return
        seen_lists.add(key)
        if w > 0 and h > 0:
            shapes.append({"type": "rect", "geometry": (x, y, w, h)})

    def _add_point(x: int, y: int) -> None:
        key = ("point", x, y)
        if key in seen_lists:
            return
        seen_lists.add(key)
        shapes.append({"type": "point", "geometry": (x, y)})

    def _visit_call(node: ast.Call) -> bool:
        """Process a function call. Returns True if handled (skip children)."""
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name == "crop_image" and node.args:
            # crop_image([x1, y1, x2, y2])
            vals = _try_extract_int_list(node.args[0])
            if vals and _is_bbox_list(vals):
                _add_rect_xyxy(*vals)
                return True

        elif func_name == "locate" and len(node.args) >= 4:
            # locate(x1, y1, x2, y2)
            coords = [_try_eval_int(a) for a in node.args[:4]]
            if all(c is not None for c in coords) and _is_bbox_list(coords):
                _add_rect_xyxy(*coords)
                return True

        elif func_name == "bbox" and len(node.args) >= 4:
            # bbox(x, y, w, h)
            coords = [_try_eval_int(a) for a in node.args[:4]]
            if all(c is not None for c in coords):
                _add_rect_xywh(*coords)
                return True

        elif func_name in ("click_at", "point") and len(node.args) >= 2:
            x = _try_eval_int(node.args[0])
            y = _try_eval_int(node.args[1])
            if (
                x is not None
                and y is not None
                and _is_grid_coord(x)
                and _is_grid_coord(y)
            ):
                _add_point(x, y)
                return True

        elif func_name == "scroll":
            # scroll(coordinate=[x,y], ...) — check keyword args
            for kw in node.keywords:
                if kw.arg == "coordinate":
                    vals = _try_extract_int_list(kw.value)
                    if vals and _is_point_list(vals):
                        _add_point(*vals)
            # Also check positional first arg
            if node.args:
                vals = _try_extract_int_list(node.args[0])
                if vals and _is_point_list(vals):
                    _add_point(*vals)
            return True

        return False

    def _visit_list(node: ast.List) -> None:
        """Check if a bare list literal is a bbox or point."""
        vals = _try_extract_int_list(node)
        if vals is None:
            return
        if _is_bbox_list(vals):
            _add_rect_xyxy(*vals)
        # Don't auto-extract 2-element lists — too many false positives

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if not _visit_call(node):
                # If call wasn't handled, check its args for list literals
                for arg in node.args:
                    if isinstance(arg, ast.List):
                        _visit_list(arg)
        elif isinstance(node, ast.Assign):
            # Check RHS for list literals (e.g. region = [50, 240, 950, 480])
            if isinstance(node.value, ast.List):
                _visit_list(node.value)

    return shapes


def extract_shapes_from_item(item: TextMessage) -> list[Shape]:
    """Extract preview shapes from a TextItem (code or tool call metadata).

    Handles:
    - Completed messages with inline code (parses item.text)
    - Client execution requests (python code from metadata)
    - click_at / scroll from request metadata
    """
    shapes: list[Shape] = []
    request = item.request if isinstance(item.request, dict) else {}
    metadata = item.metadata or {}

    # Client-side execution item with metadata arguments
    if request.get("execution") == "client":
        focus_mode = request.get("focus_mode", "")
        arguments = metadata.get("arguments", {})

        if focus_mode == "python":
            code = arguments.get("code", "")
            if code:
                shapes.extend(extract_shapes_from_code(code))

        elif focus_mode == "click_at":
            x = arguments.get("x", 0)
            y = arguments.get("y", 0)
            if _is_grid_coord(x) and _is_grid_coord(y):
                shapes.append({"type": "point", "geometry": (int(x), int(y))})

        elif focus_mode == "scroll":
            coord = arguments.get("coordinate", [500, 500])
            if isinstance(coord, list) and len(coord) == 2:
                x, y = int(coord[0]), int(coord[1])
                if _is_grid_coord(x) and _is_grid_coord(y):
                    shapes.append({"type": "point", "geometry": (x, y)})

        return shapes

    # Completed assistant message — parse text for inline code
    text = str(item.text or "")
    if text.strip():
        shapes.extend(extract_shapes_from_code(text))

    return shapes
