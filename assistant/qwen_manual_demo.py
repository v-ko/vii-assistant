"""Standalone script to probe Qwen2.5-VL manual multimodal conditioning.

The script downloads the specified checkpoint, runs the vision encoder to
produce image embeddings, scatters them into the token stream alongside a text
prompt, and executes the decoder to generate logits. The resulting logits are
compared with the model's regular forward pass to confirm parity.

Example usage:
    python -m assistant.qwen_manual_demo --model-id Qwen/Qwen2.5-VL-3B-Instruct
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


@dataclass
class DemoConfig:
    model_id: str
    device: str
    image_size: int
    precision: str | None


def main() -> None:
    config = _parse_args()
    device = torch.device(config.device)
    dtype = _select_dtype(device, config.precision)

    print(f"Loading processor and model from {config.model_id} ...")
    processor = AutoProcessor.from_pretrained(config.model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_id, torch_dtype=dtype
    )
    model.to(device)
    model.eval()

    image = _build_gradient_image(config.image_size)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": "Describe the dominant colors in this gradient.",
                },
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
    inputs = _move_to_device(inputs, device)

    forward_inputs = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }

    with torch.no_grad():
        model.rope_deltas = None
        manual_logits = _manual_forward(model, inputs)

        model.rope_deltas = None
        reference = model(**forward_inputs)
        reference_logits = reference.logits

    max_diff = (manual_logits - reference_logits).abs().max().item()
    print(
        f"Max absolute difference between manual and reference logits: {max_diff:.6e}"
    )


def _parse_args() -> DemoConfig:
    parser = argparse.ArgumentParser(
        description="Manually probe Qwen2.5-VL multimodal context assembly"
    )
    parser.add_argument(
        "--model-id",
        default=os.getenv("QWEN_VL_MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct"),
        help="HF hub identifier for the checkpoint",
    )
    parser.add_argument(
        "--device",
        default=os.getenv(
            "QWEN_VL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
        ),
        help="Inference device (e.g. cuda, cpu, cuda:1)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=int(os.getenv("QWEN_VL_TEST_IMAGE_SIZE", "224")),
        help="Width/height of synthetic gradient image",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp16", "fp32"),
        default=os.getenv("QWEN_VL_PRECISION", "auto"),
        help="Optional dtype override",
    )
    args = parser.parse_args()
    return DemoConfig(
        model_id=args.model_id,
        device=args.device,
        image_size=args.image_size,
        precision=None if args.precision == "auto" else args.precision,
    )


def _select_dtype(device: torch.device, precision: str | None) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision == "fp32":
        return torch.float32
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def _build_gradient_image(size: int) -> Image.Image:
    gradient = np.linspace(0, 255, num=size * size * 3, dtype=np.uint8)
    gradient = gradient.reshape(size, size, 3)
    return Image.fromarray(gradient, mode="RGB")


def _move_to_device(data: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in data.items()
    }


def _manual_forward(
    model: Qwen2_5_VLForConditionalGeneration, inputs: Dict[str, Any]
) -> torch.Tensor:
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs.get("pixel_values")
    pixel_values_videos = inputs.get("pixel_values_videos")
    image_grid_thw = inputs.get("image_grid_thw")
    video_grid_thw = inputs.get("video_grid_thw")

    inputs_embeds = model.model.embed_tokens(input_ids)

    if pixel_values is not None:
        pixel_values = pixel_values.to(_infer_visual_dtype(model))
        image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
        image_token_mask = input_ids == model.config.image_token_id
        _scatter_modal_embeds(inputs_embeds, image_embeds, image_token_mask)

    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.to(_infer_visual_dtype(model))
        video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw)
        video_token_mask = input_ids == model.config.video_token_id
        _scatter_modal_embeds(inputs_embeds, video_embeds, video_token_mask)

    position_ids, rope_deltas = model.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
    )
    model.rope_deltas = rope_deltas

    decoder_outputs = model.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=None,
        inputs_embeds=inputs_embeds,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )

    hidden_states = decoder_outputs.last_hidden_state
    return model.lm_head(hidden_states)


def _scatter_modal_embeds(
    inputs_embeds: torch.Tensor,
    modal_embeds: torch.Tensor,
    token_mask: torch.Tensor,
) -> None:
    modal_tokens = token_mask.sum().item()
    if modal_tokens == 0:
        return
    assert (
        modal_embeds.shape[0] == modal_tokens
    ), f"Expected {modal_tokens} vision features, got {modal_embeds.shape[0]}"
    mask = token_mask.unsqueeze(-1).expand_as(inputs_embeds)
    modal_embeds = modal_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds.masked_scatter_(mask, modal_embeds)


def _infer_visual_dtype(model: Qwen2_5_VLForConditionalGeneration) -> torch.dtype:
    for param in model.visual.parameters():
        return param.dtype
    return model.model.embed_tokens.weight.dtype


if __name__ == "__main__":
    main()
