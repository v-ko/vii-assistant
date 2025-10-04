from __future__ import annotations

import math
import re
from typing import List, Optional, Tuple

from assistant.util import get_logger

log = get_logger(__name__)

# Mirror defaults from qwen_vl_utils/src/qwen_vl_utils/vision_process.py
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28  # 3136
MAX_PIXELS = 16384 * 28 * 28  # 12,845,056
MAX_RATIO = 200

# Ollama-specific constraints for Qwen2.5-VL preprocessing
MAX_PIXELS_OLLAMA = 1_000_000


def _round_by_factor(number: float, factor: int) -> int:
    return int(round(number / factor) * factor)


def _ceil_by_factor(number: float, factor: int) -> int:
    return int(math.ceil(number / factor) * factor)


def _floor_by_factor(number: float, factor: int) -> int:
    return int(math.floor(number / factor) * factor)


def smart_resize_ollama(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    max_pixels: int = MAX_PIXELS_OLLAMA,
    strictly_less: bool = True,
) -> Tuple[int, int]:
    """
    Ollama's Qwen2.5-VL policy:
      - preserve aspect ratio
      - scale if needed so that area < 1e6 (strictly less by default)
      - round each dim to nearest multiple of 28
      - if rounding up would violate the area cap, round down
    Returns (new_h, new_w) in pixels (multiples of `factor`).
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid size {width}x{height}")
    ratio = max(height, width) / min(height, width)
    if ratio > MAX_RATIO:
        raise ValueError(f"absolute aspect ratio must be < {MAX_RATIO}, got {ratio}")

    area = height * width
    target_area = max_pixels - (1 if strictly_less else 0)

    # scale down if area >= target (for strictly_less) or > (for <=)
    if area > target_area or (strictly_less and area == target_area):
        s = math.sqrt(target_area / area)
    else:
        s = 1.0

    h_f = height * s
    w_f = width * s

    # nearest multiples of `factor`
    h_floor = _floor_by_factor(h_f, factor)
    h_ceil = _ceil_by_factor(h_f, factor)
    w_floor = _floor_by_factor(w_f, factor)
    w_ceil = _ceil_by_factor(w_f, factor)

    def ok(h: int, w: int) -> bool:
        a = h * w
        return a < max_pixels if strictly_less else a <= max_pixels

    candidates = []
    for hh in (h_floor, h_ceil):
        for ww in (w_floor, w_ceil):
            if hh >= factor and ww >= factor and ok(hh, ww):
                err = (hh - h_f) ** 2 + (ww - w_f) ** 2
                candidates.append((err, hh, ww))

    if candidates:
        _, h_bar, w_bar = min(candidates, key=lambda t: t[0])
        return int(h_bar), int(w_bar)

    # if none fit, step down in the dimension that least distorts aspect
    h_bar, w_bar = h_floor, w_floor
    while not ok(h_bar, w_bar):
        reduce_h = (h_bar / max(h_f, 1e-6)) > (w_bar / max(w_f, 1e-6))
        if reduce_h and h_bar > factor:
            h_bar -= factor
        elif w_bar > factor:
            w_bar -= factor
        elif h_bar > factor:
            h_bar -= factor
        else:
            break

    h_bar = max(h_bar, factor)
    w_bar = max(w_bar, factor)
    return int(h_bar), int(w_bar)


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> Tuple[int, int]:
    """
    Reimplementation of qwen_vl_utils.smart_resize(height, width).

    Rules:
      1) h' and w' are divisible by `factor`
      2) h'*w' is within [min_pixels, max_pixels]
      3) Aspect ratio is approximately preserved

    Returns: (new_height, new_width)
    Raises:
      ValueError if the absolute aspect ratio exceeds MAX_RATIO or if dims invalid.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid input size {width}x{height}")

    ratio = max(height, width) / min(height, width)
    if ratio > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {ratio}"
        )

    # First snap to multiples of factor
    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))

    area_bar = h_bar * w_bar
    area = height * width

    if area_bar > max_pixels:
        # Downscale: use floor so we do not exceed
        # max constraint after snapping
        beta = math.sqrt(area / max_pixels)
        h_bar = max(factor, _floor_by_factor(height / beta, factor))
        w_bar = max(factor, _floor_by_factor(width / beta, factor))
    elif area_bar < min_pixels:
        # Upscale: use ceil so we do not undershoot
        # min constraint after snapping
        beta = math.sqrt(min_pixels / area)
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)

    # Final safety (very rare off-by-one due to rounding)
    h_bar = max(factor, h_bar)
    w_bar = max(factor, w_bar)

    return int(h_bar), int(w_bar)


# Accept floats in bbox values (Qwen often emits decimals)
_bbox_pattern = re.compile(
    (
        r'["\']?bbox_2d["\']?\s*:\s*\['
        r"\s*(?P<x1>-?\d+(?:\.\d+)?)\s*,\s*(?P<y1>-?\d+(?:\.\d+)?)"
        r"\s*,\s*(?P<x2>-?\d+(?:\.\d+)?)\s*,\s*(?P<y2>-?\d+(?:\.\d+)?)\s*\]"
    ),
    re.IGNORECASE | re.DOTALL,
)


def parse_qwen25_bboxes(text: str) -> List[Tuple[float, float, float, float]]:
    """
    Parse all bbox_2d occurrences from model output text.

    Returns a list of tuples (x1, y1, x2, y2) as floats.
    """
    bboxes: List[Tuple[float, float, float, float]] = []
    for m in _bbox_pattern.finditer(text):
        x1 = float(m.group("x1"))
        y1 = float(m.group("y1"))
        x2 = float(m.group("x2"))
        y2 = float(m.group("y2"))
        bboxes.append((x1, y1, x2, y2))
    return bboxes


def _clip(val: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(val))))


def scale_bbox_from_resized_to_orig(
    bbox_xyxy: Tuple[float, float, float, float],
    input_w: int,
    input_h: int,
    orig_w: int,
    orig_h: int,
    *,
    coords_are_qwen_grid: bool = True,
    qwen_grid_size: float = 1000.0,
) -> Tuple[int, int, int, int]:
    """
    Map a bbox from the model's processed-image space back to original pixels.

    If coords_are_qwen_grid=True, bbox values are in Qwen's 0..1000 grid and
    are first converted to resized-image pixel units.
    """
    if input_w <= 0 or input_h <= 0 or orig_w <= 0 or orig_h <= 0:
        raise ValueError("Invalid dimensions for scaling")

    x1, y1, x2, y2 = bbox_xyxy
    log.debug(f"Qwen bbox (raw): [{x1}, {y1}, {x2}, {y2}]")

    # Normalize order
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    # Convert from 0..1000 grid to resized-image pixels if needed
    if coords_are_qwen_grid:
        gx = input_w / qwen_grid_size
        gy = input_h / qwen_grid_size
        x1, x2 = x1 * gx, x2 * gx
        y1, y2 = y1 * gy, y2 * gy
        log.debug(f"Bbox in resized pixels: [{x1}, {y1}, {x2}, {y2}]")

    # Linear back-mapping to original pixels
    sx = orig_w / float(input_w)
    sy = orig_h / float(input_h)

    x1_o = x1 * sx
    y1_o = y1 * sy
    x2_o = x2 * sx
    y2_o = y2 * sy
    log.debug(f"Scaled to original space (float): " f"[{x1_o}, {y1_o}, {x2_o}, {y2_o}]")

    # Clip to bounds and produce integer xywh
    final_x1 = _clip(x1_o, 0, orig_w)
    final_y1 = _clip(y1_o, 0, orig_h)
    final_x2 = _clip(x2_o, 0, orig_w)
    final_y2 = _clip(y2_o, 0, orig_h)

    w = max(0, final_x2 - final_x1)
    h = max(0, final_y2 - final_y1)

    log.debug(f"Final (x, y, w, h): [{final_x1}, {final_y1}, {w}, {h}]")
    return final_x1, final_y1, w, h


def extract_qwen25_shapes(
    text: str,
    orig_w: Optional[int],
    orig_h: Optional[int],
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> List[dict]:
    """
    End-to-end extraction for Qwen2.5-VL bbox_2d outputs:
      - parse bbox_2d arrays from text
      - compute the resized (model-input) dimensions using smart_resize
      - map bboxes back to original image pixel space
      - return shapes in { "type": "rect", "geometry": (x, y, w, h) }
    """
    log.debug(f"Extracting Qwen shapes for image size {orig_w}x{orig_h}")
    if not orig_w or not orig_h:
        log.warning("Original image size not provided for scaling.")
        return []

    # 1) Parse bboxes from text
    bboxes_xyxy = parse_qwen25_bboxes(text)
    if not bboxes_xyxy:
        log.debug("No bboxes found in text.")
        return []
    log.debug(f"Parsed {len(bboxes_xyxy)} bboxes: {bboxes_xyxy}")

    # 2) Compute resized dims that qwen-vl-utils would produce client-side
    input_h, input_w = smart_resize(
        height=orig_h,
        width=orig_w,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    log.debug(f"Resized image dimensions: {input_w}x{input_h}")

    # 3) Scale bboxes back to original coordinates and produce rect shapes
    shapes: List[dict] = []
    for xyxy in bboxes_xyxy:
        x, y, w, h = scale_bbox_from_resized_to_orig(
            xyxy,
            input_w,
            input_h,
            orig_w,
            orig_h,
            coords_are_qwen_grid=True,  # Qwen emits 0..1000 by default
            qwen_grid_size=1000.0,
        )
        if w <= 0 or h <= 0:
            log.warning(f"Skipping degenerate bbox with size {w}x{h}")
            continue
        shapes.append({"type": "rect", "geometry": (x, y, w, h)})

    return shapes


def extract_qwen25_shapes_ollama_policy(
    text: str,
    orig_w: Optional[int],
    orig_h: Optional[int],
) -> List[dict]:
    """
    Qwen2.5-VL bbox_2d extraction using Ollama's resize policy:

      - parse bbox_2d arrays from text
      - compute the resized (model-input) dimensions using smart_resize_ollama
      - map bboxes back to original image pixel space
      - return shapes in { "type": "rect", "geometry": (x, y, w, h) }
    """
    log.debug(f"[Ollama policy] Extracting Qwen shapes for image {orig_w}x{orig_h}")
    if not orig_w or not orig_h:
        log.warning("Original image size not provided for scaling.")
        return []

    bboxes_xyxy = parse_qwen25_bboxes(text)
    if not bboxes_xyxy:
        log.debug("No bboxes found in text.")
        return []
    log.debug(f"Parsed {len(bboxes_xyxy)} bboxes: {bboxes_xyxy}")

    # Use Ollama-compatible resize for back-projection grid
    input_h, input_w = smart_resize_ollama(height=orig_h, width=orig_w)
    log.debug(f"[Ollama policy] Resized dims: {input_w}x{input_h}")

    shapes: List[dict] = []
    for xyxy in bboxes_xyxy:
        x, y, w, h = scale_bbox_from_resized_to_orig(
            xyxy,
            input_w,
            input_h,
            orig_w,
            orig_h,
            coords_are_qwen_grid=True,
            qwen_grid_size=1000.0,
        )
        if w <= 0 or h <= 0:
            log.warning(f"Skipping degenerate bbox with size {w}x{h}")
            continue
        shapes.append({"type": "rect", "geometry": (x, y, w, h)})

    return shapes
