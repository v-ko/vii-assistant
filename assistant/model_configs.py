from transformers import (
    Qwen3_5ForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

MODEL_SPECS = {
    "qwen3_vl_30b_a3b": {
        "id": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "class": Qwen3VLForConditionalGeneration,
    },
    "qwen3_vl_4b": {
        "id": "Qwen/Qwen3-VL-4B-Instruct",
        "class": Qwen3VLForConditionalGeneration,
    },
    "qwen3_5_4b": {
        "id": "Qwen/Qwen3.5-4B",
        "class": Qwen3_5ForConditionalGeneration,
    },
}

# ACTIVE_MODEL_KEY = "qwen3_vl_30b_a3b"
# ACTIVE_MODEL_KEY = "qwen3_5_4b"
ACTIVE_MODEL_KEY = "qwen3_vl_4b"

MODEL_ID = MODEL_SPECS[ACTIVE_MODEL_KEY]["id"]
MODEL_CLASS = MODEL_SPECS[ACTIVE_MODEL_KEY]["class"]

MODEL_CLASS_MAP = {spec["id"]: spec["class"] for spec in MODEL_SPECS.values()}
