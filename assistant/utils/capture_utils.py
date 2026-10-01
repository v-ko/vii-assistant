from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Optional

from PySide6.QtCore import SLOT, QObject, QRect, QUrl, Slot
from PySide6.QtDBus import (
    QDBusConnection,
    QDBusMessage,
    QDBusPendingCallWatcher,
)
from PySide6.QtGui import QGuiApplication, QImage, QPixmap, QScreen

log = logging.getLogger(__name__)

_PORTAL_SERVICE = "org.freedesktop.portal.Desktop"
_PORTAL_PATH = "/org/freedesktop/portal/desktop"
_REQUEST_IFACE = "org.freedesktop.portal.Request"
# Generous: the first call may wait on the user answering a permission dialog
_PORTAL_TIMEOUT_S = 60.0


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


def use_portal_screenshots() -> bool:
    """Qt can't grab the desktop on Wayland; VII_SCREENSHOT_BACKEND=portal|qt overrides."""
    backend = os.environ.get("VII_SCREENSHOT_BACKEND", "").strip().lower()
    if backend:
        return backend == "portal"
    return os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"


def take_screenshot(screen: QScreen | None) -> Optional[QPixmap]:
    """Synchronous grab (Qt backend only). Prefer `grab_screen` in async code."""
    if screen is None:
        return None
    if use_portal_screenshots():
        log.error("Synchronous screenshots are unavailable with the portal backend")
        return None
    try:
        return screen.grabWindow(0)
    except Exception as exc:  # noqa: BLE001
        log.error("Error taking screenshot: %s", exc)
        return None


async def grab_screen(screen: QScreen | None) -> Optional[QPixmap]:
    if screen is None:
        return None
    shots = await grab_screens([screen])
    return shots.get(screen.name())


async def grab_screens(screens: list[QScreen]) -> dict[str, QPixmap]:
    """Capture each screen, keyed by screen name. Missing keys mean failure."""
    if not use_portal_screenshots():
        return {s.name(): s.grabWindow(0) for s in screens}

    desktop = await _portal_screenshot()
    if desktop is None:
        return {}

    # The portal returns the whole virtual desktop; crop per screen
    virtual = QGuiApplication.primaryScreen().virtualGeometry()
    scale = desktop.width() / virtual.width()
    shots: dict[str, QPixmap] = {}
    for screen in screens:
        g = screen.geometry().translated(-virtual.topLeft())
        rect = QRect(
            round(g.x() * scale),
            round(g.y() * scale),
            round(g.width() * scale),
            round(g.height() * scale),
        )
        shots[screen.name()] = QPixmap.fromImage(desktop.copy(rect))
    return shots


class _PortalRequest(QObject):
    def __init__(self, future: asyncio.Future) -> None:
        super().__init__()
        self._future = future

    @Slot("uint", "QVariantMap")
    def on_response(self, code: int, results: dict) -> None:
        if not self._future.done():
            self._future.set_result((code, results))

    def on_call_finished(self, watcher: QDBusPendingCallWatcher) -> None:
        if watcher.isError() and not self._future.done():
            self._future.set_exception(RuntimeError(watcher.error().message()))


async def _portal_screenshot() -> Optional[QImage]:
    """Non-interactive org.freedesktop.portal.Screenshot of the whole desktop."""
    bus = QDBusConnection.sessionBus()
    token = f"vii_{uuid.uuid4().hex}"
    sender = bus.baseService().lstrip(":").replace(".", "_")
    request_path = f"{_PORTAL_PATH}/request/{sender}/{token}"

    future: asyncio.Future = asyncio.get_running_loop().create_future()
    request = _PortalRequest(future)
    signal_args = (_PORTAL_SERVICE, request_path, _REQUEST_IFACE, "Response")
    slot = SLOT("on_response(uint,QVariantMap)")
    # Subscribe before calling so the Response signal can't be missed
    bus.connect(*signal_args, request, slot)
    try:
        msg = QDBusMessage.createMethodCall(
            _PORTAL_SERVICE,
            _PORTAL_PATH,
            "org.freedesktop.portal.Screenshot",
            "Screenshot",
        )
        msg.setArguments(["", {"handle_token": token, "interactive": False}])
        watcher = QDBusPendingCallWatcher(bus.asyncCall(msg), request)
        watcher.finished.connect(request.on_call_finished)

        code, results = await asyncio.wait_for(future, timeout=_PORTAL_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001
        log.error("Portal screenshot failed: %s", exc)
        return None
    finally:
        bus.disconnect(*signal_args, request, slot)

    if code != 0:
        log.error("Portal screenshot denied or cancelled (response=%s)", code)
        return None

    path = QUrl(results.get("uri", "")).toLocalFile()
    image = QImage(path)
    try:
        os.remove(path)
    except OSError as exc:
        log.warning("Could not remove portal screenshot %s: %s", path, exc)
    if image.isNull():
        log.error("Portal screenshot could not be loaded from %s", path)
        return None
    return image
