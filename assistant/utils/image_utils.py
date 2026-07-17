"""Qt image conversion utilities (formerly in sivkit.platform.qt_widgets.utils)."""

from __future__ import annotations

from io import BytesIO

from PIL import Image
from PySide6.QtCore import QBuffer, QIODevice
from PySide6.QtGui import QImage, QPixmap


def qpixmap_to_pil(pixmap: QPixmap) -> Image.Image:
    """Convert a QPixmap to a PIL Image."""
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    pixmap.save(buf, "PNG")
    buf.close()
    return Image.open(BytesIO(buf.data().data())).convert("RGB")


def qimage_to_pil(image: QImage) -> Image.Image:
    """Convert a QImage to a PIL Image."""
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buf, "PNG")
    buf.close()
    return Image.open(BytesIO(buf.data().data())).convert("RGB")
