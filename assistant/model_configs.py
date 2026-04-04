from transformers import (
    Qwen3_5ForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

MODEL_SPECS = {
    "qwen3_vl_4b": {
        "id": "Qwen/Qwen3-VL-4B-Instruct",
        "class": Qwen3VLForConditionalGeneration,
        "display_name": "Qwen3-VL 4B",
    },
    "qwen3_vl_30b_a3b": {
        "id": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "class": Qwen3VLForConditionalGeneration,
        "display_name": "Qwen3-VL 30B (A3B)",
    },
    "qwen3_5_4b": {
        "id": "Qwen/Qwen3.5-4B",
        "class": Qwen3_5ForConditionalGeneration,
        "display_name": "Qwen3.5 4B",
    },
}

DEFAULT_MODEL_KEY = "qwen3_vl_4b"

AVAILABLE_MODELS: dict[str, str] = {"none": "No model"}
AVAILABLE_MODELS.update({k: v["display_name"] for k, v in MODEL_SPECS.items()})

MODEL_CLASS_MAP = {spec["id"]: spec["class"] for spec in MODEL_SPECS.values()}
