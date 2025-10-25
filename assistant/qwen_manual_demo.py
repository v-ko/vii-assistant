from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast

import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration as QwenModel

from assistant.image_ops import resize_like_preprocessor
from assistant.model_configs import MODEL_CLASS_MAP, MODEL_ID

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Ordered variety of image sizes to stress resizing & positional embedding logic
IMAGE_SIZES = [32, 224, 512, 1024]
# None => auto select (fp16 on CUDA, fp32 on CPU). Explicit options: "bf16", "fp16", "fp32"
PRECISION: str | None = None
MAX_NEW_TOKENS = 128


@dataclass
class DemoConfig:
    model_id: str
    device: str
    image_sizes: List[int]
    precision: str | None
    max_new_tokens: int


@dataclass
class PreparedInputs:
    input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    pixel_values: Optional[torch.Tensor]
    pixel_values_videos: Optional[torch.Tensor]
    image_grid_thw: Optional[torch.Tensor]
    video_grid_thw: Optional[torch.Tensor]


def main() -> None:
    config = DemoConfig(
        model_id=MODEL_ID,
        device=DEVICE,
        image_sizes=IMAGE_SIZES,
        precision=PRECISION,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    device = torch.device(config.device)
    dtype = _select_dtype(device, config.precision)

    print(f"Loading processor and model from {config.model_id} ...")
    processor = AutoProcessor.from_pretrained(config.model_id)
    modelClass = MODEL_CLASS_MAP[config.model_id]
    model = modelClass.from_pretrained(config.model_id, torch_dtype=dtype)
    model.to(device)  # type: ignore[call-arg]
    model.eval()

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
    for size in config.image_sizes:
        print("\n" + "=" * 10 + f" Processing image {size}x{size} " + "=" * 10)
        image = _build_gradient_image(size)
        manual_resized_image, resize_meta = resize_like_preprocessor(
            image, processor.image_processor
        )

        inputs = processor(
            text=[prompt], images=[image], return_tensors="pt", padding=True
        )
        inputs = _move_to_device(inputs, device)

        manual_pixel_values = processor.image_processor(
            manual_resized_image,
            do_resize=False,
            return_tensors="pt",
        )["pixel_values"].to(device)

        _assert_same_pixels(manual_pixel_values, inputs["pixel_values"])
        inputs["pixel_values"] = manual_pixel_values

        prepared = _prepare_inputs(inputs, manual_pixel_values)
        model_kwargs = _build_model_kwargs(prepared)

        with torch.no_grad():
            _reset_rope_state(model)
            manual_logits = _manual_forward(model, prepared)

            _reset_rope_state(model)
            reference = model(return_dict=True, **model_kwargs)
            reference_logits = reference.logits

        max_diff = (manual_logits - reference_logits).abs().max().item()
        print(
            "Max absolute difference between manual and reference logits:"
            f" {max_diff:.6e}"
        )
        print(
            "Manual resize produced"
            f" {resize_meta['width']}x{resize_meta['height']} pixels; processor tensor"
            f" shape: {tuple(int(dim) for dim in manual_pixel_values.shape)}"
        )

        print("Generating model response...")
        _reset_rope_state(model)
        generation = model.generate(
            **model_kwargs,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
        )
        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = generation[:, prompt_length:]
        decoded = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        print("Model output:")
        print(decoded.strip())


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


def _optional_tensor(data: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    value = data.get(key)
    if value is None:
        return None
    return cast(torch.Tensor, value)


def _prepare_inputs(raw: Dict[str, Any], pixel_values: torch.Tensor) -> PreparedInputs:
    return PreparedInputs(
        input_ids=cast(torch.Tensor, raw["input_ids"]),
        attention_mask=_optional_tensor(raw, "attention_mask"),
        pixel_values=pixel_values,
        pixel_values_videos=_optional_tensor(raw, "pixel_values_videos"),
        image_grid_thw=_optional_tensor(raw, "image_grid_thw"),
        video_grid_thw=_optional_tensor(raw, "video_grid_thw"),
    )


def _build_model_kwargs(prepared: PreparedInputs) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"input_ids": prepared.input_ids}
    if prepared.attention_mask is not None:
        kwargs["attention_mask"] = prepared.attention_mask
    if prepared.pixel_values is not None:
        kwargs["pixel_values"] = prepared.pixel_values
    if prepared.pixel_values_videos is not None:
        kwargs["pixel_values_videos"] = prepared.pixel_values_videos
    if prepared.image_grid_thw is not None:
        kwargs["image_grid_thw"] = prepared.image_grid_thw
    if prepared.video_grid_thw is not None:
        kwargs["video_grid_thw"] = prepared.video_grid_thw
    return kwargs


def _reset_rope_state(model: QwenModel) -> None:
    setattr(model, "rope_deltas", None)  # type: ignore[attr-defined]


def _move_to_device(data: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in data.items()
    }


def _manual_forward(model: QwenModel, prepared: PreparedInputs) -> torch.Tensor:
    input_ids = prepared.input_ids
    attention_mask = prepared.attention_mask
    pixel_values = prepared.pixel_values
    pixel_values_videos = prepared.pixel_values_videos
    image_grid_thw = prepared.image_grid_thw
    video_grid_thw = prepared.video_grid_thw

    # embed_tokens is an nn.Embedding but not perfectly typed in generated stubs
    embed_tokens = cast(nn.Embedding, model.model.embed_tokens)
    inputs_embeds = embed_tokens(input_ids)

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

    position_ids, rope_deltas = cast(Any, model).get_rope_index(  # type: ignore[attr-defined]
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


def _infer_visual_dtype(model: QwenModel) -> torch.dtype:
    first_visual = next(model.visual.parameters(), None)
    if first_visual is not None:
        return first_visual.dtype
    embed_tokens = getattr(model.model, "embed_tokens", None)
    if embed_tokens is not None and hasattr(embed_tokens, "parameters"):
        first_text = next(embed_tokens.parameters(), None)
        if first_text is not None:
            return first_text.dtype
    return torch.float32


def _assert_same_pixels(reference: torch.Tensor, candidate: torch.Tensor) -> None:
    torch.testing.assert_close(reference, candidate)


if __name__ == "__main__":
    main()
