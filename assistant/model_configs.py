from unittest.mock import DEFAULT

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

MODEL_CLASS_MAP = {
    "Qwen/Qwen2.5-VL-3B-Instruct": Qwen2_5_VLForConditionalGeneration,
    "Qwen/Qwen3-VL-4B-Instruct": Qwen3VLForConditionalGeneration,
}
