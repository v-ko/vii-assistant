import requests
from typing import Optional, Union, List, cast
import io
from PIL import Image

from assistant.config import MODEL_EXTRACTOR_MAP
from assistant.services.base_client import BaseClient
from assistant.util import Shape, get_logger

log = get_logger(__name__)


class OllamaClient(BaseClient):
    """Client for interacting with the Ollama API."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        super().__init__(base_url)
        self.api_url = f"{base_url}/api"
        self._last_model: Optional[str] = None

    def call_vlm(
        self,
        model: str,
        prompt: str,
        image_data: Optional[Union[bytes, io.BytesIO, Image.Image]] = None,
    ) -> str:
        """Call the Vision Language Model with a prompt and optional image.

        Args:
            model: The name of the model to use (e.g., "moondream")
            prompt: The text prompt to send to the model
            image_data: Optional image data as bytes, BytesIO, or PIL Image

        Returns:
            The model's response as a string
        """
        # Prepare the request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            # Limit output to prevent hanging
            "options": {
                "num_predict": 1000
            },
        }
        # Remember last-used model for extractor routing
        self._last_model = model

        # If image data is provided, encode it as base64
        if image_data is not None:
            try:
                base64_image = self.encode_image(image_data)
                payload["images"] = [base64_image]
            except Exception as e:
                print(f"Error processing image data: {e}")
                return f"Error processing image: {e}"

        # Make the API request
        try:
            response = requests.post(
                f"{self.api_url}/generate",
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                result = response.json()
                print("Result:", result)
                return result.get("response", "No response from model")
            else:
                return f"Error: {response.status_code} - {response.text}"

        except Exception as e:
            return f"Error calling Ollama API: {e}"

    def is_available(self) -> bool:
        """Check if the Ollama service is available."""
        try:
            response = requests.get(f"{self.api_url}/tags")
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """List available models."""
        try:
            response = requests.get(f"{self.api_url}/tags")
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            return []
        except Exception:
            return []

    def extract_shapes(
        self,
        text: str,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> List[Shape]:
        """
        Route to a model-specific extractor via config.MODEL_EXTRACTOR_MAP.
        Falls back to BaseClient.extract_shapes when no mapping exists.
        """
        log.info(
            f"Extracting shapes via routing. "
            f"Image: {image_width}x{image_height}"
        )

        # Require image dims for bbox-based extractors
        if image_width is None or image_height is None:
            return super().extract_shapes(text, image_width, image_height)

        model_name = getattr(self, "_last_model", "") or ""
        # This client is the Ollama backend
        extractor_map = MODEL_EXTRACTOR_MAP.get("ollama", {})
        extractor = extractor_map.get(model_name)

        if extractor:
            # Cast return to Shape list
            return cast(
                List[Shape],
                extractor(text, image_width, image_height),
            )

        return super().extract_shapes(text, image_width, image_height)
