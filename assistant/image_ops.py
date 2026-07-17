"""Image preprocessing helpers shared across inference code."""

from __future__ import annotations

from typing import Protocol, TypedDict

from PIL import Image
from qwen_vl_utils import smart_resize


class SupportsVisionResize(Protocol):
    patch_size: int
    min_pixels: int
    max_pixels: int


class ResizeMetadata(TypedDict):
    width: int
    height: int
    paste_x: int
    paste_y: int
    scale: float
    src_width: int
    src_height: int


def resize_like_preprocessor(
    image: Image.Image,
    image_processor: SupportsVisionResize,
) -> tuple[Image.Image, ResizeMetadata]:
    resized_height, resized_width = smart_resize(
        image.height,
        image.width,
        factor=image_processor.patch_size,
        min_pixels=image_processor.min_pixels,
        max_pixels=image_processor.max_pixels,
    )
    resized_image = image.resize((resized_width, resized_height))
    src_w, src_h = image.size
    scale = min(resized_width / src_w, resized_height / src_h)
    return resized_image, ResizeMetadata(
        width=resized_width,
        height=resized_height,
        paste_x=0,
        paste_y=0,
        scale=scale,
        src_width=src_w,
        src_height=src_h,
    )


def resize_to_target(
    image: Image.Image,
    target_w: int,
    target_h: int,
) -> tuple[Image.Image, ResizeMetadata]:
    """Resize image to exact target dimensions, fitting within the frame
    while preserving aspect ratio (no distortion). The image is scaled to
    fill as much of the target as possible, then center-padded with black
    on the shorter axis.
    """
    src_w, src_h = image.size
    scale = min(target_w / src_w, target_h / src_h)
    scaled_w = round(src_w * scale)
    scaled_h = round(src_h * scale)
    scaled = image.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    paste_x = (target_w - scaled_w) // 2
    paste_y = (target_h - scaled_h) // 2
    canvas.paste(scaled, (paste_x, paste_y))
    return canvas, ResizeMetadata(
        width=target_w,
        height=target_h,
        paste_x=paste_x,
        paste_y=paste_y,
        scale=scale,
        src_width=src_w,
        src_height=src_h,
    )


def scale_qwen_bbox_xyxy(
    bbox_xyxy: tuple[int, int, int, int],
    *,
    input_w: int,
    input_h: int,
    orig_w: int,
    orig_h: int,
    coords_are_qwen_grid: bool = True,
) -> tuple[int, int, int, int]:
    """Scale a bbox defined on Qwen's 0..1000 grid (or already resized coords) back to original image/screen.

    Expects bbox in (x1,y1,x2,y2). Returns scaled (x1,y1,x2,y2) in original coordinate space.
    """
    x1, y1, x2, y2 = bbox_xyxy
    if coords_are_qwen_grid:
        rx1 = int(x1 / 1000 * input_w)
        ry1 = int(y1 / 1000 * input_h)
        rx2 = int(x2 / 1000 * input_w)
        ry2 = int(y2 / 1000 * input_h)
    else:
        rx1, ry1, rx2, ry2 = x1, y1, x2, y2
    sx = int(rx1 / input_w * orig_w)
    sy = int(ry1 / input_h * orig_h)
    ex = int(rx2 / input_w * orig_w)
    ey = int(ry2 / input_h * orig_h)
    return sx, sy, ex, ey


def scale_qwen_point(
    point: tuple[int, int],
    *,
    input_w: int,
    input_h: int,
    orig_w: int,
    orig_h: int,
    coords_are_qwen_grid: bool = True,
) -> tuple[int, int]:
    x, y = point
    if coords_are_qwen_grid:
        rx = int(x / 1000 * input_w)
        ry = int(y / 1000 * input_h)
    else:
        rx, ry = x, y
    sx = int(rx / input_w * orig_w)
    sy = int(ry / input_h * orig_h)
    return sx, sy


def qwen_grid_to_original_image(
    bbox_xyxy: tuple[int, int, int, int],
    meta: ResizeMetadata,
) -> tuple[float, float, float, float]:
    """Map a bbox from Qwen's 0-1000 grid on the padded canvas back to
    original image pixel coordinates.

    Returns (x1, y1, x2, y2) in original image pixel space.
    """
    x1, y1, x2, y2 = bbox_xyxy
    canvas_w, canvas_h = meta["width"], meta["height"]
    # 0-1000 grid → canvas pixels
    cx1 = x1 / 1000 * canvas_w
    cy1 = y1 / 1000 * canvas_h
    cx2 = x2 / 1000 * canvas_w
    cy2 = y2 / 1000 * canvas_h
    # Undo center-padding
    cx1 -= meta["paste_x"]
    cy1 -= meta["paste_y"]
    cx2 -= meta["paste_x"]
    cy2 -= meta["paste_y"]
    # Undo scale
    scale = meta["scale"]
    ox1 = cx1 / scale
    oy1 = cy1 / scale
    ox2 = cx2 / scale
    oy2 = cy2 / scale
    return ox1, oy1, ox2, oy2


def qwen_point_to_original_image(
    point: tuple[int, int],
    meta: ResizeMetadata,
) -> tuple[float, float]:
    """Map a point from Qwen's 0-1000 grid on the padded canvas back to
    original image pixel coordinates."""
    x, y = point
    canvas_w, canvas_h = meta["width"], meta["height"]
    cx = x / 1000 * canvas_w - meta["paste_x"]
    cy = y / 1000 * canvas_h - meta["paste_y"]
    scale = meta["scale"]
    return cx / scale, cy / scale
