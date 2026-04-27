"""Probe each model's image preprocessor to discover output resolutions.

Feeds checkerboard images (with red outline) at various sizes, all at 16:9 AR,
through each model's AutoProcessor and records the output resolution.
Saves input images to tmp/preprocessor_probe/ for visual inspection.

Usage:
    python -m scripts.probe_preprocessor_resolutions
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from assistant.model_configs import MODEL_SPECS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TARGET_AR = 1920 / 1080  # 16:9
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "tmp" / "preprocessor_probe"

# Standard-ish 16:9 probe sizes (width x height), capped around 2500px per edge.
PROBE_SIZES: list[tuple[int, int]] = [
    (398, 224),
    (480, 270),
    (640, 360),
    (800, 450),
    (960, 540),
    (1024, 576),
    (1280, 720),
    (1366, 768),
    (1600, 900),
    (1920, 1080),
    (2048, 1152),
    (2560, 1440),
]


def create_checkerboard(height: int, width: int, square_size: int = 20) -> Image.Image:
    x_tiles = width // square_size + 2
    y_tiles = height // square_size + 2
    board = 127.0 * np.kron(
        [[1, 0] * (x_tiles // 2), [0, 1] * (x_tiles // 2)] * (y_tiles // 2),
        np.ones((square_size, square_size)),
    )
    board += 128
    board = board[:height, :width]
    rgb = np.stack([board, board, board], axis=2).astype(np.uint8)
    img = Image.fromarray(rgb)

    draw = ImageDraw.Draw(img)
    for i in range(3):
        draw.rectangle([i, i, width - 1 - i, height - 1 - i], outline=(255, 0, 0))
    return img


def _build_prompt(processor) -> str:
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": "describe"}],
        },
    ]
    try:
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        log.warning("apply_chat_template failed (%s), using plain text prompt", e)
        return "describe"


def _resolve_size_from_output(processor, out: dict) -> tuple[int, int] | None:
    """Extract (height, width) in pixels from processor output dict."""
    pv = out.get("pixel_values")
    if pv is None:
        log.warning("No pixel_values in output. Keys: %s", list(out.keys()))
        return None

    # Route 1: image_grid_thw present (Qwen-family)
    if "image_grid_thw" in out:
        grid = out["image_grid_thw"]
        if isinstance(grid, torch.Tensor) and grid.numel() >= 3:
            _t, h_patches, w_patches = (int(v) for v in grid[0].tolist())
            pp = getattr(processor, "image_processor", processor)
            patch_size = getattr(pp, "patch_size", None)
            if patch_size is None:
                log.error("image_grid_thw present but no patch_size on processor")
                return None
            h_px = int(h_patches) * patch_size
            w_px = int(w_patches) * patch_size
            log.debug(
                "grid thw=(%s,%s,%s) patch=%s -> %sx%s",
                _t,
                h_patches,
                w_patches,
                patch_size,
                w_px,
                h_px,
            )
            return h_px, w_px

    # Route 2: image_position_ids present (Gemma4-family)
    # Shape: (batch, num_patches, 2) where dim2 = [x, y] (width, height), -1 = padding
    if "image_position_ids" in out:
        pos = out["image_position_ids"]
        if isinstance(pos, torch.Tensor) and pos.ndim == 3 and pos.shape[2] >= 2:
            pos_2d = pos[0]  # (num_patches, 2)
            valid = pos_2d[:, 0] >= 0  # filter out -1 padding
            if valid.any():
                x_pos = pos_2d[valid, 0]  # width dimension
                y_pos = pos_2d[valid, 1]  # height dimension
                w_patches = int(x_pos.max().item()) + 1
                h_patches = int(y_pos.max().item()) + 1
                pp = getattr(processor, "image_processor", processor)
                patch_size = getattr(pp, "patch_size", None)
                if patch_size is None:
                    log.error(
                        "image_position_ids present but no patch_size on processor"
                    )
                    return None
                h_px = h_patches * patch_size
                w_px = w_patches * patch_size
                log.debug(
                    "image_position_ids w_patches=%s h_patches=%s patch=%s -> %sx%s",
                    w_patches,
                    h_patches,
                    patch_size,
                    w_px,
                    h_px,
                )
                return h_px, w_px

    # Route 2: standard (B, C, H, W) tensor
    if isinstance(pv, torch.Tensor):
        shape = pv.shape
        log.debug("pixel_values tensor shape: %s", shape)
        if len(shape) == 4:
            return int(shape[2]), int(shape[3])
        if len(shape) == 5:
            return int(shape[3]), int(shape[4])
    elif isinstance(pv, (list, tuple)) and pv:
        t = pv[0]
        if isinstance(t, torch.Tensor):
            shape = t.shape
            log.debug("pixel_values[0] tensor shape: %s", shape)
            if len(shape) == 4:
                return int(shape[2]), int(shape[3])
            if len(shape) == 3 and shape[0] > 3:
                # (num_patches, C) — no spatial info extractable
                log.warning(
                    "pixel_values is flat patches (%s) with no image_grid_thw", shape
                )

    log.warning(
        "Could not determine output size from pixel_values. "
        "type=%s shape=%s all_keys=%s",
        type(pv).__name__,
        getattr(pv, "shape", None)
        or (f"list[{len(pv)}]" if isinstance(pv, (list, tuple)) else "?"),
        list(out.keys()),
    )
    # Dump first element info if it's a list
    if isinstance(pv, (list, tuple)) and pv:
        first = pv[0]
        log.warning(
            "  pv[0]: type=%s shape=%s",
            type(first).__name__,
            getattr(first, "shape", "?"),
        )
    return None


def get_processor_output_size(
    processor,
    prompt: str,
    image: Image.Image,
    extra_kwargs: dict | None = None,
) -> tuple[int, int] | None:
    """Feed image through processor, return (height, width) of resulting pixels."""
    kw = extra_kwargs or {}
    try:
        out = processor(text=[prompt], images=[image], return_tensors="pt", **kw)
    except Exception as e1:
        log.debug("processor(text=, images=[img]) failed: %s", e1)
        try:
            out = processor(images=[image], return_tensors="pt", **kw)
        except Exception as e2:
            log.debug("processor(images=[img]) failed: %s", e2)
            try:
                out = processor(images=image, return_tensors="pt", **kw)
            except Exception as e3:
                log.error("All processor call variants failed: %s", e3)
                return None

    return _resolve_size_from_output(processor, out)


def _sweep_sizes(
    processor,
    prompt: str,
    model_dir: Path,
    label: str,
    extra_kwargs: dict | None = None,
) -> dict:
    """Probe all PROBE_SIZES through the processor and return mapping + unique outputs."""
    results: dict[tuple[int, int], tuple[int, int]] = {}
    seen_outputs: set[tuple[int, int]] = set()
    total = len(PROBE_SIZES)

    for idx, (w, h) in enumerate(PROBE_SIZES):
        img = create_checkerboard(h, w)
        out_size = get_processor_output_size(
            processor, prompt, img, extra_kwargs=extra_kwargs
        )

        if out_size is None:
            log.warning("[%s %d/%d] %sx%s -> FAILED", label, idx + 1, total, w, h)
            continue

        out_wh = (out_size[1], out_size[0])  # (w, h)
        results[(w, h)] = out_wh

        if out_wh not in seen_outputs:
            seen_outputs.add(out_wh)
            fname = f"{label}_in_{w}x{h}_out_{out_wh[0]}x{out_wh[1]}.png"
            img.save(model_dir / fname)
            log.info(
                "[%s %d/%d] %sx%s -> %sx%s  (NEW)",
                label,
                idx + 1,
                total,
                w,
                h,
                out_wh[0],
                out_wh[1],
            )
        elif idx % 20 == 0:
            log.info(
                "[%s %d/%d] %sx%s -> %sx%s",
                label,
                idx + 1,
                total,
                w,
                h,
                out_wh[0],
                out_wh[1],
            )

    unique_outputs = sorted(seen_outputs)
    log.info("%s — unique output resolutions (%d):", label, len(unique_outputs))
    for ow, oh in unique_outputs:
        log.info("  %sx%s", ow, oh)

    return {
        "mapping": [
            {"input": [iw, ih], "output": [ow, oh]}
            for (iw, ih), (ow, oh) in sorted(results.items())
        ],
        "unique_outputs": [[ow, oh] for ow, oh in unique_outputs],
    }


# Gemma4 supports configurable vision token budgets
_GEMMA4_TOKEN_BUDGETS = [70, 140, 280, 560, 1120]


def probe_model(model_key: str, spec: dict) -> dict:
    model_id = spec["id"]
    log.info("=" * 60)
    log.info("Probing: %s (%s)", model_key, model_id)
    log.info("=" * 60)

    t0 = time.monotonic()
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        log.error("Failed to load processor: %s", e)
        return {}
    log.info(
        "Processor loaded in %.1fs: %s", time.monotonic() - t0, type(processor).__name__
    )

    pp = getattr(processor, "image_processor", None)
    if pp:
        log.info(
            "image_processor attrs: patch_size=%s min_pixels=%s max_pixels=%s merge_size=%s "
            "max_soft_tokens=%s pooling_kernel_size=%s",
            getattr(pp, "patch_size", "?"),
            getattr(pp, "min_pixels", "?"),
            getattr(pp, "max_pixels", "?"),
            getattr(pp, "merge_size", "?"),
            getattr(pp, "max_soft_tokens", "?"),
            getattr(pp, "pooling_kernel_size", "?"),
        )
    else:
        log.warning("No image_processor attribute on processor")

    prompt = _build_prompt(processor)
    log.info("Prompt (first 80 chars): %s", prompt[:80])

    # Sanity check
    test_img = create_checkerboard(224, 398)
    log.info("Sanity check with 398x224 image...")
    test_result = get_processor_output_size(processor, prompt, test_img)
    if test_result is None:
        log.error("Sanity check FAILED — skipping this model")
        return {}
    log.info("Sanity check OK: output %sx%s (h,w)", test_result[0], test_result[1])

    model_dir = OUTPUT_DIR / model_key
    model_dir.mkdir(parents=True, exist_ok=True)

    base_attrs = (
        {
            "patch_size": getattr(pp, "patch_size", None),
            "min_pixels": getattr(pp, "min_pixels", None),
            "max_pixels": getattr(pp, "max_pixels", None),
            "merge_size": getattr(pp, "merge_size", None),
            "max_soft_tokens": getattr(pp, "max_soft_tokens", None),
            "pooling_kernel_size": getattr(pp, "pooling_kernel_size", None),
        }
        if pp
        else None
    )

    # Check if this is a Gemma4-like model with configurable token budgets
    has_token_budget = pp and hasattr(pp, "max_soft_tokens")

    if has_token_budget:
        log.info("Gemma4-style model: sweeping token budgets %s", _GEMMA4_TOKEN_BUDGETS)
        token_budget_sweeps = {}
        for budget in _GEMMA4_TOKEN_BUDGETS:
            label = f"tokens_{budget}"
            sweep = _sweep_sizes(
                processor,
                prompt,
                model_dir,
                label,
                extra_kwargs={"max_soft_tokens": budget},
            )
            token_budget_sweeps[str(budget)] = sweep

        return {
            "model_id": model_id,
            "processor_class": type(processor).__name__,
            "image_processor_attrs": base_attrs,
            "variable_token_budget": True,
            "supported_budgets": _GEMMA4_TOKEN_BUDGETS,
            "default_budget": getattr(pp, "max_soft_tokens", 280),
            "by_token_budget": token_budget_sweeps,
        }
    else:
        sweep = _sweep_sizes(processor, prompt, model_dir, "default")
        return {
            "model_id": model_id,
            "processor_class": type(processor).__name__,
            "image_processor_attrs": base_attrs,
            "variable_token_budget": False,
            **sweep,
        }


RESULTS_JSON_PATH = OUTPUT_DIR / "probe_results.json"


def main():
    if OUTPUT_DIR.exists():
        log.info("Cleaning previous results: %s", OUTPUT_DIR)
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    vision_specs = {k: v for k, v in MODEL_SPECS.items() if v.get("vision")}
    skipped = [k for k in MODEL_SPECS if k not in vision_specs]
    if skipped:
        log.info("Skipping non-vision models: %s", skipped)

    import transformers

    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "transformers_version": transformers.__version__,
        "probe_sizes": [[w, h] for w, h in PROBE_SIZES],
        "models": {},
    }

    for key, spec in vision_specs.items():
        report["models"][key] = probe_model(key, spec)

    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(report, f, indent=2)
    log.info("JSON report saved to: %s", RESULTS_JSON_PATH)


if __name__ == "__main__":
    main()
