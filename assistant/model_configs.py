from transformers import (
    AutoModelForMultimodalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

# Allowed (width, height) resolutions per model — populated via
# scripts/probe_preprocessor_resolutions.py.  Keep sorted by area ascending.
# These are what the preprocessor actually produces for 16:9-ish inputs.

# Qwen3-VL: dynamic, rounds each edge to nearest multiple of 16.
# All Qwen3-VL-based models share the same scheme.
_QWEN3_VL_RESOLUTIONS: list[tuple[int, int]] = [
    (384, 224),
    (480, 256),
    (640, 352),
    (800, 448),
    (960, 544),
    (1024, 576),
    (1280, 704),
    (1376, 768),
    (1600, 896),
    (1920, 1088),
    (2048, 1152),
    (2560, 1440),
]

# Gemma4: resolution depends on max_soft_tokens (token budget).
# Each budget yields a single fixed resolution for 16:9 inputs.
# patch_size=16, pooling_kernel_size=3, side_mult=48
# Mapping: budget -> (width, height)
_GEMMA4_BUDGET_RESOLUTIONS: dict[int, tuple[int, int]] = {
    70: (528, 288),
    140: (720, 384),
    280: (1056, 576),
    560: (1488, 816),
    1120: (2112, 1200),
}
_GEMMA4_RESOLUTIONS: list[tuple[int, int]] = sorted(
    _GEMMA4_BUDGET_RESOLUTIONS.values(), key=lambda wh: wh[0] * wh[1]
)

MODEL_SPECS = {
    "qwen3_vl_4b": {
        "id": "Qwen/Qwen3-VL-4B-Instruct",
        "class": Qwen3VLForConditionalGeneration,
        "display_name": "Qwen3-VL 4B",
        "vision": True,
        "allowed_resolutions": _QWEN3_VL_RESOLUTIONS,
        "default_resolution": (1920, 1088),
    },
    "qwen3_vl_30b_a3b": {
        "id": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "class": Qwen3VLForConditionalGeneration,
        "display_name": "Qwen3-VL 30B (A3B)",
        "vision": True,
        "allowed_resolutions": _QWEN3_VL_RESOLUTIONS,
        "default_resolution": (1920, 1088),
    },
    "gui_owl_1_5_4b": {
        "id": "mPLUG/GUI-Owl-1.5-4B-Instruct",
        "class": Qwen3VLForConditionalGeneration,
        "display_name": "GUI-Owl 1.5 4B",
        "vision": True,
        "allowed_resolutions": _QWEN3_VL_RESOLUTIONS,
        "default_resolution": (1920, 1088),
    },
    "qwen3_5_4b": {
        "id": "Qwen/Qwen3.5-4B",
        "class": Qwen3_5ForConditionalGeneration,
        "display_name": "Qwen3.5 4B",
        "vision": True,
        "allowed_resolutions": _QWEN3_VL_RESOLUTIONS,
        "default_resolution": (1920, 1088),
        "chat_template_params": {
            "enable_thinking": False,
        },
    },
    "gemma4_e4b": {
        "id": "google/gemma-4-E4B-it",
        "class": AutoModelForMultimodalLM,
        "display_name": "Gemma 4 E4B",
        "vision": True,
        "allowed_resolutions": _GEMMA4_RESOLUTIONS,
        "default_resolution": (1056, 576),  # budget=280
        "chat_template_params": {
            "enable_thinking": False,
        },
    },
}

DEFAULT_MODEL_KEY = "qwen3_vl_4b"

AVAILABLE_MODELS: dict[str, str] = {"none": "No model"}
AVAILABLE_MODELS.update({k: v["display_name"] for k, v in MODEL_SPECS.items()})

MODEL_CLASS_MAP = {spec["id"]: spec["class"] for spec in MODEL_SPECS.values()}


def get_resolution_for_model(
    model_key: str,
    override: tuple[int, int] | None = None,
) -> tuple[int, int]:
    spec = MODEL_SPECS.get(model_key)
    if not spec or not spec.get("vision"):
        raise ValueError(f"Model {model_key!r} is not a vision model or does not exist")

    allowed = spec.get("allowed_resolutions", [])
    if override is not None:
        if allowed and override not in allowed:
            raise ValueError(
                f"Resolution {override[0]}x{override[1]} not in allowed set for {model_key}. "
                f"Allowed: {allowed}"
            )
        return override

    default = spec.get("default_resolution")
    if default is None:
        raise ValueError(
            f"No default_resolution configured for {model_key!r}. "
            f"Run scripts/probe_preprocessor_resolutions.py and fill in the config."
        )
    return default
