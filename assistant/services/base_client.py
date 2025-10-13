import base64
import io
from typing import List, Optional, Union

from PIL import Image

from assistant.services import qwen25_preprocess
from assistant.util import Shape, extract_coordinates

# Centralized client/model defaults (previously in config.py)
DEFAULT_OLLAMA_URL = "http://desk:11434"
DEFAULT_VLLM_URL = "http://desk:8000"
DEFAULT_MODEL = "qwen2.5vl"  # Primary vision model in use


# Client configuration mapping backends to available models
CLIENT_CONFIG = {
    "ollama": [
        "moondream",
        "gemma3",
        "qwen2.5vl",
    ],
    "vllm": ["osunlp/UGround-V1-2B"],
}
DEFAULT_CLIENT_TYPE = "ollama:moondream"


# Hardcoded model-to-extractor key routing per backend (no regex)
# Keys in the inner dict must exactly match model names in CLIENT_CONFIG.
MODEL_EXTRACTOR_MAP = {
    "ollama": {
        "qwen2.5vl": qwen25_preprocess.extract_qwen25_shapes_ollama_policy,
    },
    "vllm": {
        # Add vLLM model-specific extractors here if needed
    },
}


class BaseClient:
    """Base client for interacting with vision language model APIs."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"  # Default API URL pattern, can be overridden

    def call_vlm(
        self,
        model: str,
        prompt: str,
        image_data: Optional[Union[bytes, io.BytesIO, Image.Image]] = None,
    ) -> str:
        """Call the Vision Language Model with a prompt and optional image.

        This method should be implemented by subclasses.

        Args:
            model: The name of the model to use
            prompt: The text prompt to send to the model
            image_data: Optional image data as bytes, BytesIO, or PIL Image

        Returns:
            The model's response as a string
        """
        raise NotImplementedError("Subclasses must implement call_vlm")

    def format_response(self, prompt: str, response: str) -> str:
        """Format the response using the template.

        Template:
        {{ if .Prompt }} Question: {{ .Prompt }}
        {{ end }} Answer: {{ .Response }}
        """
        if prompt:
            return f"Question: {prompt}\n\nAnswer: {response}"
        else:
            return f"Answer: {response}"

    def is_available(self) -> bool:
        """Check if the service is available.

        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement is_available")

    def list_models(self) -> List[str]:
        """List available models.

        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement list_models")

    def encode_image(self, image_data: Union[bytes, io.BytesIO, Image.Image]) -> str:
        """Encode image data to base64 string.

        Args:
            image_data: Image data as bytes, BytesIO, or PIL Image

        Returns:
            Base64 encoded string
        """
        try:
            if isinstance(image_data, Image.Image):
                # Convert PIL Image to bytes
                img_byte_arr = io.BytesIO()
                image_data.save(img_byte_arr, format="PNG")
                return base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
            elif isinstance(image_data, io.BytesIO):
                # BytesIO object
                return base64.b64encode(image_data.getvalue()).decode("utf-8")
            else:
                # Raw bytes
                return base64.b64encode(image_data).decode("utf-8")
        except Exception as e:
            print(f"Error encoding image: {e}")
            raise

    def extract_shapes(
        self,
        text: str,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> List[Shape]:
        """Extract shapes from model response text.

        This default implementation uses the extract_coordinates function from util.py.
        Subclasses can override this method to provide model-specific extraction logic.

        Args:
            text: The text to extract shapes from
            image_width: Optional image width for coordinate scaling
            image_height: Optional image height for coordinate scaling

        Returns:
            A list of shapes
        """
        return extract_coordinates(text)
