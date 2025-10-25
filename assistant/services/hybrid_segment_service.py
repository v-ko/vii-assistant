from __future__ import annotations

import re
from typing import Optional, TypedDict

from fusion import get_logger
from fusion.util.rectangle import Rectangle
from PIL import Image
from PySide6.QtGui import QGuiApplication

from assistant.facade import facade
from assistant.image_ops import resize_like_preprocessor, scale_qwen_bbox_xyxy
from assistant.inference.context import ContentType, ContextItem
from assistant.util import Shape

log = get_logger(__name__)


def capture(
    x: int, y: int, width: int, height: int
) -> Rectangle:  # placeholder for future screenshot logic
    return Rectangle(x, y, width, height)


_SEG_LINE = re.compile(
    r"^\s*(?:"  # start line, optional assignment prefix
    r"(output\.[A-Za-z0-9_.]+)\s*=\s*"  # group(1) lhs if present
    r")?(Rectangle|capture)\s*\((?P<args>[^)]*)\)\s*$"
)


class SegmentOutput(TypedDict, total=False):
    bbox: Rectangle | None
    # image may eventually be a PIL Image; for now we also allow a Rectangle region
    image: Rectangle | Image.Image | None


class HybridSegmentService:
    """Parses last assistant text output for Rectangle/capture lines and updates overlay.

    Coordinate conversion: assumes emitted Rectangle/capture coordinates are in Qwen 0..1000 grid.
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
        output_model = self._parse_output(last_ai_text)
        rects: list[Rectangle] = []
        if output_model.get("bbox"):
            rects.append(output_model["bbox"])  # type: ignore[index]
        # For now image field stores rect too until actual screenshot integrated
        img_val = output_model.get("image")
        if isinstance(
            img_val, Rectangle
        ):  # treat image region same as bbox for overlay
            rects.append(img_val)
        if not rects:
            # Clear overlay on no shapes
            try:
                facade.qt_app.overlay.set_shapes([])
            except Exception:
                pass
            return
        shapes = self._convert_to_shapes(rects)
        try:
            facade.qt_app.overlay.set_shapes(shapes)
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
        out: SegmentOutput = {"bbox": None, "image": None}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            m = _SEG_LINE.match(line)
            if not m:
                continue
            lhs = m.group(1)  # may be None for bare call
            tool = m.group(2)
            args_raw = m.group("args")
            parts = [p.strip() for p in args_raw.split(",") if p.strip()]
            if len(parts) != 4:
                log.error(
                    "Segment parse error (arg count != 4)"
                    + (f" lhs={lhs}" if lhs else "")
                    + f" tool={tool} raw='{args_raw}'"
                )
                continue
            try:
                x, y, w, h = [int(p) for p in parts]
            except ValueError:
                log.error(
                    "Segment parse error (non-integer args)"
                    + (f" lhs={lhs}" if lhs else "")
                    + f" tool={tool} parts={parts}"
                )
                continue
            rect = Rectangle(x, y, w, h) if tool == "Rectangle" else capture(x, y, w, h)
            if lhs:
                if lhs not in ("output.bbox", "output.image"):
                    log.error(f"Segment parse ignore (invalid lhs root): {lhs}")
                    continue
                if lhs.endswith("bbox"):
                    out["bbox"] = rect
                else:
                    out["image"] = rect
            else:
                # Bare invocation policy
                if tool == "Rectangle" and out.get("bbox") is None:
                    out["bbox"] = rect
                elif tool == "capture" and out.get("image") is None:
                    out["image"] = rect
        return out

    def _convert_to_shapes(self, rects: list[Rectangle]) -> list[Shape]:
        # Acquire screen geometry (watched screen per settings)
        try:
            screen = facade.current_watched_screen()
        except Exception:
            screen = QGuiApplication.primaryScreen()
        if not screen:
            return []
        geo = screen.geometry()
        orig_w = int(geo.width())
        orig_h = int(geo.height())
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
                        break

        if input_w == orig_w and input_h == orig_h:
            # Fallback derive resized dims to mimic model policy for scaling
            class _DummyProcessor:
                patch_size = 28
                min_pixels = 4 * patch_size * patch_size
                max_pixels = 16384 * patch_size * patch_size

            synthetic = Image.new("RGB", (orig_w, orig_h))
            _, meta = resize_like_preprocessor(synthetic, _DummyProcessor())
            input_w = meta["width"]
            input_h = meta["height"]
        shapes: list[Shape] = []

        for r in rects:
            # Interpret x,y,width,height as Qwen grid xywh -> convert to xyxy first
            # Qwen lines expected may already be xywh or xyxy; spec given: Rectangle(x,y,w,h)
            x1 = r.x()
            y1 = r.y()
            x2 = r.right()
            y2 = r.bottom()
            # Cast to int to satisfy typing contract (Rectangle may yield float via right()/bottom() in future changes)
            sx, sy, ex, ey = scale_qwen_bbox_xyxy(
                (int(x1), int(y1), int(x2), int(y2)),
                input_w=input_w,
                input_h=input_h,
                orig_w=orig_w,
                orig_h=orig_h,
                coords_are_qwen_grid=True,
            )
            w = max(0, ex - sx)
            h = max(0, ey - sy)
            # Use fusion Rectangle for final shape geometry
            final = Rectangle(sx, sy, w, h)
            gx, gy, gw, gh = final.as_tuple()
            # Cast to int for Shape geometry contract
            rect_shape: Shape = {
                "type": "rect",
                "geometry": (int(gx), int(gy), int(gw), int(gh)),
                "color": "#FF0000",
            }
            shapes.append(rect_shape)
        return shapes


__all__ = ["HybridSegmentService", "capture"]
