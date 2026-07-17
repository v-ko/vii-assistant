import re
from typing import Any, Final, List, Literal, Optional, Tuple, TypedDict, Union, cast

from PIL import Image
from PySide6.QtCore import QBuffer, QByteArray, QIODevice
from PySide6.QtGui import QGuiApplication, QImage, QPixmap, QScreen


class BaseShape(TypedDict, total=False):
    """Base shape type with common optional properties."""

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


def get_screen_by_name(name: str) -> QScreen | None:
    for screen in QGuiApplication.screens():
        if screen.name() == name:
            return screen
    return None


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
        if pixmap.isNull():
            return None
        byte_array = QByteArray()
        buffer = QBuffer(byte_array)
        if not buffer.open(QIODevice.OpenModeFlag.WriteOnly):
            return None
        # Use QPixmap.save directly; pass format as str (PySide6 expects str overload).
        ok = pixmap.save(buffer, "PNG")
        buffer.close()
        if not ok:
            return None
        # Ensure we convert memoryview -> bytes before decoding (helps type checkers)
        return bytes(byte_array.toBase64().data()).decode("utf-8")
    except Exception as e:
        print(f"Error converting QPixmap to base64: {e}")
        return None


def qimage_to_pil(img: QImage) -> Image.Image:
    if img.isNull():
        raise ValueError("Cannot convert null QImage")

    qimg = img.convertToFormat(QImage.Format.Format_RGB888)
    w: Final[int] = qimg.width()
    h: Final[int] = qimg.height()
    stride: Final[int] = qimg.bytesPerLine()
    size_bytes: Final[int] = stride * h

    # QImage.bits() returns a sip.voidptr at runtime; stubs don't reflect this
    p: Any = qimg.bits()
    try:
        p.setsize(size_bytes)
        mv = memoryview(p)[:size_bytes].cast("B")
    except AttributeError:
        mv = memoryview(p.asstring(size_bytes))

    buf = bytes(mv)

    pil = Image.frombuffer("RGB", (w, h), buf, "raw", "RGB", stride, 0)
    # If qimg might be freed/modified, detach:
    # pil = pil.copy()
    return pil
