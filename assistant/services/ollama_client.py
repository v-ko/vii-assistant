import requests
from typing import Optional, Union, List
import io
from PIL import Image

from assistant.services.base_client import BaseClient


class OllamaClient(BaseClient):
    """Client for interacting with the Ollama API."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        super().__init__(base_url)
        self.api_url = f"{base_url}/api"

    def call_vlm(
        self,
        model: str,
        prompt: str,
        image_data: Optional[Union[bytes, io.BytesIO, Image.Image]] = None
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
            "options": {
                "num_predict": 1000  # Limit output to prevent hanging
            }
        }

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
                headers={"Content-Type": "application/json"})

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
        except:
            return False

    def list_models(self) -> List[str]:
        """List available models."""
        try:
            response = requests.get(f"{self.api_url}/tags")
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            return []
        except:
            return []
