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


def resize_like_processor(
    image: Image.Image, image_processor: SupportsVisionResize
) -> tuple[Image.Image, ResizeMetadata]:
    """Resize *image* the same way the Qwen processor would."""

    factor = getattr(image_processor, "patch_size", 28)
    min_pixels = getattr(image_processor, "min_pixels", 4 * factor * factor)
    max_pixels = getattr(image_processor, "max_pixels", 16384 * factor * factor)

    resized_height, resized_width = smart_resize(
        image.height,
        image.width,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    resized_image = image.resize((resized_width, resized_height))
    return resized_image, ResizeMetadata(width=resized_width, height=resized_height)
