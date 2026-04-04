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
    return resized_image, ResizeMetadata(width=resized_width, height=resized_height)


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
