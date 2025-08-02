import requests
import re
from typing import Optional, Union, List
import io
from PIL import Image

from assistant.services.base_client import BaseClient
from assistant.util import Shape, PointShape


class VLLMClient(BaseClient):
    """Client for interacting with the VLLM API via an OpenAI-style endpoint."""

    def __init__(self, base_url: str, model: str):
        super().__init__(base_url)
        self.model = model
        # OpenAI-style API endpoint
        self.api_url = f"{base_url}/v1"

    def call_vlm(
        self,
        model: str,  # This parameter is kept for compatibility with BaseClient
        prompt: str,
        image_data: Optional[Union[bytes, io.BytesIO,
                                   Image.Image]] = None) -> str:
        """Call the visual language model with a system prompt and an optional image.

        Args:
            model: Ignored, using self.model instead
            prompt: The system prompt for guiding the model
            image_data: Optional image data (bytes, BytesIO, or PIL Image)

        Returns:
            The raw model output as a string
        """
        if image_data is None:
            return "Error: Image data is required for this model"

        try:
            base64_image = self.encode_image(image_data)
        except Exception as e:
            print(f"Error processing image data: {e}")
            return f"Error processing image: {e}"

        # Format messages according to the provided template
        messages = [
            {
                "role":
                "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt  # Use the prompt parameter directly
                    },
                ],
            },
        ]

        # Prepare the payload with the formatted messages and temperature=0
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,  # Set temperature to zero as specified
            "stream": False,
            "max_tokens": 1000,  # Limit output to prevent hanging
        }

        # print(f"VLLM Request payload: {payload}")

        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"})
            print(f"VLLM Response status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                # print(f"VLLM Response: {result}")

                # Extract the response content
                content = result.get("choices",
                                     [{}])[0].get("message", {}).get(
                                         "content", "No response from model")

                print(f"VLLM Content: {content}")

                # Note: The output coordinates are in the range [0,1000)
                # and need to be scaled to the actual screen dimensions
                # This scaling would typically be done in the overlay component

                return content
            else:
                error_msg = f"Error: {response.status_code} - {response.text}"
                print(error_msg)
                return error_msg
        except Exception as e:
            error_msg = f"Error calling VLLM API: {e}"
            print(error_msg)
            return error_msg

    def is_available(self) -> bool:
        """Check if the VLLM service is available."""
        try:
            # Simple health check endpoint
            response = requests.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """List available models.

        For VLLM, we just return the configured model.
        """
        if self.is_available():
            return [self.model]
        return []

    def extract_shapes(self,
                       text: str,
                       image_width: Optional[int] = None,
                       image_height: Optional[int] = None) -> List[Shape]:
        """Extract shapes from model response text for UGround model.

        The UGround model returns coordinates in the range [0,1000),
        which need to be scaled to the actual image dimensions.

        Args:
            text: The text to extract shapes from
            image_width: Image width for coordinate scaling
            image_height: Image height for coordinate scaling

        Returns:
            A list of shapes
        """
        # First use the base implementation to extract shapes
        shapes = super().extract_shapes(text)

        # Then adjust the coordinates for UGround model if image dimensions are provided
        if image_width is not None and image_height is not None:
            for shape in shapes:
                if shape['type'] == 'point':
                    # Get the original coordinates
                    orig_x, orig_y = shape['geometry']

                    # Scale coordinates from [0,1000) range to image dimensions
                    scaled_x = int(orig_x / 1000 * image_width)
                    scaled_y = int(orig_y / 1000 * image_height)

                    # Update the shape with scaled coordinates
                    shape['geometry'] = (scaled_x, scaled_y)

        print('args:', text, image_width, image_height)
        print(f"Extracted shapes: {shapes}")
        return shapes
