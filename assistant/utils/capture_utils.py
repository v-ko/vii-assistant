from __future__ import annotations

from typing import Optional

from PySide6.QtGui import QGuiApplication, QPixmap, QScreen


def clipboard_image() -> Optional[QPixmap]:
    clipboard = QGuiApplication.clipboard()
    if clipboard is None:
        return None
    mime = clipboard.mimeData()
    if mime is None or not mime.hasImage():
        return None
    image_data = mime.imageData()
    if not image_data:
        return None
    return QPixmap(image_data)


def take_screenshot(screen: QScreen | None) -> Optional[QPixmap]:
    if screen is None:
        return None
    try:
        return screen.grabWindow(0)
    except Exception as exc:  # noqa: BLE001
        print(f"Error taking screenshot: {exc}")
        return None
