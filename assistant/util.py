import logging
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union

from PySide6.QtCore import QBuffer, QByteArray, QIODevice
from PySide6.QtGui import QPixmap


class BaseShape(TypedDict, total=False):
    """Base shape type with common properties."""

    type: str
    geometry: Any
    color: Optional[str]  # Color in hex format (e.g., '#FF0000')


class RectShape(BaseShape):
    """Rectangle shape type."""

    type: Literal["rect"]
    geometry: Tuple[int, int, int, int]  # x, y, width, height


class PointShape(BaseShape):
    """Point shape type."""

    type: Literal["point"]
    geometry: Tuple[int, int]  # x, y


Shape = Union[RectShape, PointShape]

logging.basicConfig(level=logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    logger = logging.getLogger(name)
    return logger


def encode_client_type(backend: str, model: str) -> str:
    """Encode backend and model into a client_type string.

    Args:
        backend: The backend name (e.g., "ollama", "vllm")
        model: The model name (e.g., "moondream", "osunlp/UGround-V1-2B")

    Returns:
        A client_type string in the format "backend:model"
    """
    return f"{backend}:{model}"


def decode_client_type(client_type: str) -> Tuple[str, str]:
    """Decode a client_type string into backend and model.

    Args:
        client_type: A client_type string in the format "backend:model"

    Returns:
        A tuple of (backend, model)
    """
    parts = client_type.split(":", 1)
    if len(parts) != 2:
        # Default to ollama:moondream if format is invalid
        return "ollama", "moondream"
    return parts[0], parts[1]


def extract_coordinates(text: str) -> List[Shape]:
    """Extract coordinates from text.

    Looks for patterns like (x,y) or (x, y) in the text and converts them to
    point shapes.

    Args:
        text: The text to extract coordinates from

    Returns:
        A list of shapes
    """
    shapes: List[Shape] = []

    # Pattern for (x,y) or (x, y)
    pattern = r"\((\d+)\s*,\s*(\d+)\)"
    matches = re.findall(pattern, text)

    for match in matches:
        x, y = int(match[0]), int(match[1])
        # Create a point shape
        point_shape: PointShape = {
            "type": "point",
            "geometry": (x, y),
        }
        shapes.append(point_shape)

    return shapes


def pixmap_to_base64(pixmap: QPixmap) -> Optional[str]:
    """Convert a QPixmap to a base64-encoded string.

    Args:
        pixmap: The QPixmap to convert

    Returns:
        A base64-encoded string, or None if conversion fails
    """
    try:
        # Convert QPixmap to QImage
        image = pixmap.toImage()

        # Create a byte array to store the image data
        byte_array = QByteArray()
        buffer = QBuffer(byte_array)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)

        # Save the image to the buffer in PNG format
        if not image.save(buffer, b"PNG"):
            raise RuntimeError("Failed to save image to buffer")

        # Convert to base64
        base64_data = bytes(byte_array.toBase64().data()).decode("utf-8")

        return base64_data
    except Exception as e:
        print(f"Error converting QPixmap to base64: {e}")
        return None
