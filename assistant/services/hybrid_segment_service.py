from __future__ import annotations

import json
import re
from typing import Optional, TypedDict

from fusion import get_logger
from fusion.util.rectangle import Rectangle
from PIL import Image
from PySide6.QtGui import QGuiApplication

from assistant.facade import facade
from assistant.image_ops import (
    resize_like_preprocessor,
    scale_qwen_bbox_xyxy,
    scale_qwen_point,
)
from assistant.inference.context import ContentType, ContextItem
from assistant.inference.function_interpreter import HybridFunctionInterpreter
from assistant.util import Shape

log = get_logger(__name__)

# Create module-level HFI instance
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


class SegmentOutput(TypedDict, total=False):
    bbox: list[Rectangle]
    # image may eventually be a PIL Image; for now we also allow a Rectangle region
    image: list[Rectangle] | list[Image.Image]
    points: list[tuple[int, int]]


class HybridSegmentService:
    """Parses last assistant text output for Rectangle/locate lines and updates overlay.

    Coordinate conversion: assumes emitted Rectangle/locate coordinates are in Qwen 0..1000 grid.
    """

    def __init__(self) -> None:
        pass

    # External entrypoint (invoked on context updates)
    def handle_context_change(self) -> None:
        try:
            last_ai_text = self._last_ai_text()
        except Exception:
            return
        if not last_ai_text:
            return
        # log.info(f"Processing AI text for segments: {last_ai_text[:200]}...")
        output_model = self._parse_output(last_ai_text)
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
            # log.info("No shapes found, clearing overlay")
            try:
                facade.qt_app.overlay.set_shapes([])
            except Exception:
                pass
            return
        shapes = self._convert_to_shapes(rects, points)
        log.info(f"Converted to {len(shapes)} overlay shapes")
        try:
            facade.qt_app.overlay.set_shapes(shapes)
            log.info("Successfully set overlay shapes")
        except Exception as e:
            log.error(f"Failed to set overlay shapes: {e}")

    # --- internals ---
    def _last_ai_text(self) -> Optional[str]:
        ctx_mgr = facade.context_manager
        items = list(ctx_mgr.items_sorted())
        # iterate reverse; pick first assistant-origin text item
        for item in reversed(items):
            try:
                # Treat presence of request field as marker for AI message output now
                if item.request is not None and item.content_type() is ContentType.TEXT:
                    txt = str(item.content.get("text") or "")
                    if txt.strip():
                        return txt
            except Exception:
                continue
        return None

    def _parse_output(self, text: str) -> SegmentOutput:
        out: SegmentOutput = {"bbox": [], "image": [], "points": []}

        self._parse_json_blocks(text, out)

        for line in text.splitlines():
            if not line.strip():
                continue

            parsed = hfi.parse_line(line)
            if parsed is None:
                continue

            lhs, func_name, args = parsed
            log.debug(f"Parsing segment line: lhs={lhs}, func={func_name}, args={args}")

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

        # log.info(
        #     f"Parse complete: bbox={len(out.get('bbox', []))},"
        #     f" image={len(out.get('image', []))},"
        #     f" points={len(out.get('points', []))}"
        # )
        return out

    def _parse_json_blocks(self, text: str, out: SegmentOutput) -> None:
        for m in re.finditer(r"```json\s*\n(.*?)```", text, re.DOTALL):
            raw = m.group(1).strip()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                log.error(f"JSON block parse error: {e}")
                continue

            items: list[dict] = []
            if isinstance(data, dict):
                items = [data]
            elif isinstance(data, list):
                items = [d for d in data if isinstance(d, dict)]
            else:
                log.error(f"JSON block has unexpected type: {type(data)}")
                continue

            for item in items:
                self._parse_json_item(item, out)

    def _parse_json_item(self, item: dict, out: SegmentOutput) -> None:
        bbox_val = item.get("bbox_2d") or item.get("bbox")
        if isinstance(bbox_val, list) and len(bbox_val) == 4:
            try:
                x1, y1, x2, y2 = (int(v) for v in bbox_val)
                rect = Rectangle(x1, y1, x2 - x1, y2 - y1)
                log.info(f"JSON bbox {bbox_val} [xyxy] -> Rectangle{rect.as_tuple()}")
                out["bbox"].append(rect)  # type: ignore[union-attr]
            except (ValueError, TypeError) as e:
                log.error(f"JSON bbox conversion error: {e}")

        point_val = item.get("point")
        if isinstance(point_val, list) and len(point_val) == 2:
            try:
                px, py = int(point_val[0]), int(point_val[1])
                log.info(f"JSON point [{px}, {py}]")
                out["points"].append((px, py))  # type: ignore[union-attr]
            except (ValueError, TypeError) as e:
                log.error(f"JSON point conversion error: {e}")

    def _convert_to_shapes(
        self, rects: list[Rectangle], points: list[tuple[int, int]] | None = None
    ) -> list[Shape]:
        # Acquire screen geometry (watched screen per settings)
        try:
            screen = facade.current_watched_screen()
        except Exception:
            screen = QGuiApplication.primaryScreen()
        if not screen:
            log.error("No screen available for shape conversion")
            return []
        geo = screen.geometry()
        orig_w = int(geo.width())
        orig_h = int(geo.height())
        log.info(f"Screen geometry: {orig_w}x{orig_h}")
        # Use explicit image_size metadata from the last image context item if available.
        input_w = orig_w
        input_h = orig_h
        for it in facade.context_manager.items_reversed():
            if "image" in (it.content or {}):
                sz_meta = (it.metadata or {}).get("image_size")  # type: ignore[union-attr]
                if sz_meta and isinstance(sz_meta, dict):
                    iw = sz_meta.get("width")
                    ih = sz_meta.get("height")
                    if isinstance(iw, int) and isinstance(ih, int):
                        input_w = iw
                        input_h = ih
                        log.info(f"Found image size in metadata: {input_w}x{input_h}")
                        break

        if input_w == orig_w and input_h == orig_h:
            # Fallback derive resized dims to mimic model policy for scaling
            log.info("No image metadata found, deriving resized dimensions")

            class _DummyProcessor:
                patch_size = 28
                min_pixels = 4 * patch_size * patch_size
                max_pixels = 16384 * patch_size * patch_size

            synthetic = Image.new("RGB", (orig_w, orig_h))
            _, meta = resize_like_preprocessor(synthetic, _DummyProcessor())
            input_w = meta["width"]
            input_h = meta["height"]
            log.info(f"Derived input dimensions: {input_w}x{input_h}")
        shapes: list[Shape] = []

        for r in rects:
            # Interpret x,y,width,height as Qwen grid xywh -> convert to xyxy first
            # Qwen lines expected may already be xywh or xyxy; spec given: bbox(x,y,w,h)
            x1 = r.x()
            y1 = r.y()
            x2 = r.right()
            y2 = r.bottom()
            log.info(
                f"Processing rectangle: x1={x1}, y1={y1}, x2={x2}, y2={y2} (from Qwen"
                " grid)"
            )
            # Cast to int to satisfy typing contract (Rectangle may yield float via right()/bottom() in future changes)
            sx, sy, ex, ey = scale_qwen_bbox_xyxy(
                (int(x1), int(y1), int(x2), int(y2)),
                input_w=input_w,
                input_h=input_h,
                orig_w=orig_w,
                orig_h=orig_h,
                coords_are_qwen_grid=True,
            )
            log.info(f"Scaled to screen: sx={sx}, sy={sy}, ex={ex}, ey={ey}")
            w = max(0, ex - sx)
            h = max(0, ey - sy)
            # Use fusion Rectangle for final shape geometry
            final = Rectangle(sx, sy, w, h)
            gx, gy, gw, gh = final.as_tuple()
            log.info(f"Final shape geometry: x={gx}, y={gy}, w={gw}, h={gh}")
            # Cast to int for Shape geometry contract
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
                orig_w=orig_w,
                orig_h=orig_h,
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
