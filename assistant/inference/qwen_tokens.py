"""Utilities for compiling contexts into Qwen inputs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, cast

from PIL import Image
from transformers import AutoProcessor

from assistant.image_ops import resize_like_preprocessor
from assistant.inference.context import ContextManager, MessageBatch

logger = logging.getLogger(__name__)


@dataclass
class CompiledQwenContext:
    prompt: str
    messages: List[Dict[str, Any]]
    images: List[Image.Image]
    processor_inputs: Dict[str, Any]
    resize_metadata: List[Dict[str, int]]


def compile_qwen_context(
    context_manager: ContextManager,
    processor: AutoProcessor,
    add_generation_prompt: bool = True,
) -> CompiledQwenContext:
    batch: MessageBatch = context_manager.items_as_qwen_chat_messages()
    messages, images = batch.messages, batch.images
    logger.info(
        "Compiling context for Qwen: messages=%d images=%d",
        len(messages),
        len(images),
    )
    resized_images: List[Image.Image] = []
    resize_meta: List[Dict[str, int]] = []
    # Use processor.image_processor via Any to satisfy static typing
    _pp: Any = processor
    for idx, image in enumerate(images):
        resized, meta = resize_like_preprocessor(image, _pp.image_processor)  # type: ignore[attr-defined]
        if resized.size != image.size:
            logger.warning(
                f"Preprocessor resized image {idx} from {image.width}x{image.height} to"
                f" {resized.width}x{resized.height}"
            )
        resized_images.append(resized)
        resize_meta.append(cast(Dict[str, int], meta))

    prompt = cast(Any, processor).apply_chat_template(  # type: ignore[attr-defined]
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    processor_kwargs: Dict[str, Any] = {
        "text": [prompt],
        "return_tensors": "pt",
        "padding": True,
    }
    if resized_images:
        processor_kwargs["images"] = resized_images
    processor_outputs = cast(Any, processor)(**processor_kwargs)  # type: ignore[misc]

    return CompiledQwenContext(
        prompt=prompt,
        messages=messages,
        images=resized_images,
        processor_inputs=dict(processor_outputs),
        resize_metadata=resize_meta,
    )
