"""Utilities for compiling contexts into Qwen inputs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

from PIL import Image
from transformers import AutoProcessor

from assistant.inference.context import ContextManager, MessageBatch

logger = logging.getLogger(__name__)


@dataclass
class CompiledQwenContext:
    prompt: str
    messages: list[dict[str, Any]]
    images: list[Image.Image]
    processor_inputs: dict[str, Any]
    resize_metadata: list[dict[str, int]]


def compile_qwen_context(
    context_manager: ContextManager,
    processor: AutoProcessor,
    add_generation_prompt: bool = True,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> CompiledQwenContext:
    batch: MessageBatch = context_manager.items_as_qwen_chat_messages()
    messages, images = batch.messages, batch.images
    logger.info(
        "Compiling context for Qwen: messages=%d images=%d",
        len(messages),
        len(images),
    )
    # Images arrive already resized to the target resolution by the client.
    # No further resize — pass through as-is.
    resize_meta: list[dict[str, int]] = [
        {"width": img.width, "height": img.height} for img in images
    ]

    prompt = cast(Any, processor).apply_chat_template(  # type: ignore[attr-defined]
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **(chat_template_kwargs or {}),
    )
    processor_kwargs: dict[str, Any] = {
        "text": [prompt],
        "return_tensors": "pt",
        "padding": True,
    }
    if images:
        processor_kwargs["images"] = images
    processor_outputs = cast(Any, processor)(**processor_kwargs)  # type: ignore[misc]

    return CompiledQwenContext(
        prompt=prompt,
        messages=messages,
        images=images,
        processor_inputs=dict(processor_outputs),
        resize_metadata=resize_meta,
    )
