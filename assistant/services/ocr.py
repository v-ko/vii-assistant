from __future__ import annotations

import io
from typing import Callable, Optional

import pytesseract
from PIL import Image
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QThread, Signal
from PySide6.QtGui import QPixmap


class _OCRWorker(QThread):
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, pixmap: QPixmap, lang: str = "eng"):
        super().__init__()
        self._pixmap = pixmap
        self._lang = lang

    def run(self):  # type: ignore[override]
        try:
            ba = QByteArray()
            buffer = QBuffer(ba)
            buffer.open(QIODevice.OpenModeFlag.WriteOnly)
            try:
                self._pixmap.save(buffer, "PNG")
            finally:
                buffer.close()
            raw = ba.data()
            pil_image = Image.open(io.BytesIO(bytes(raw)))
            text = pytesseract.image_to_string(pil_image, lang=self._lang)
            self.finished.emit((text.strip()) or "<No text recognized>")
        except Exception as e:  # pragma: no cover
            self.error.emit(f"Error during OCR: {e}")


def start_ocr(
    pixmap: QPixmap,
    *,
    on_finished: Callable[[str], None],
    on_error: Optional[Callable[[str], None]] = None,
    lang: str = "eng",
) -> QThread:
    worker = _OCRWorker(pixmap, lang=lang)
    worker.finished.connect(on_finished)
    if on_error:
        worker.error.connect(on_error)
    else:  # default: forward error as finished message prefixed
        worker.error.connect(lambda msg: on_finished(f"Error: {msg}"))
    worker.start()
    return worker


def ocr_sync(pixmap: QPixmap, lang: str = "eng") -> str:
    ba = QByteArray()
    buffer = QBuffer(ba)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    try:
        pixmap.save(buffer, "PNG")
    finally:
        buffer.close()
    raw = ba.data()
    pil_image = Image.open(io.BytesIO(bytes(raw)))
    text = pytesseract.image_to_string(pil_image, lang=lang)
    return text.strip() or "<No text recognized>"
