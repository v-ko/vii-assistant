import logging
from typing import Optional
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QByteArray, QBuffer, QIODevice

logging.basicConfig(level=logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    logger = logging.getLogger(name)
    return logger


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
        image.save(buffer, b"PNG")

        # Convert to base64
        base64_data = bytes(byte_array.toBase64().data()).decode("utf-8")

        return base64_data
    except Exception as e:
        print(f"Error converting QPixmap to base64: {e}")
        return None
