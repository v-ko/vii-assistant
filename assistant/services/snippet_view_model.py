"""SnippetViewModel — manages fullscreen overlays for screen region selection.

Activated via the `snippet` command. Calls the show_snippet_overlays action
which captures screenshots and creates view states. The view model reacts
to view state changes by creating/destroying actual QML overlay windows.

On completion or cancellation, calls hide_snippet_overlays to clean up.
"""

from __future__ import annotations

import logging
from pathlib import Path

import shiboken6
from PySide6.QtCore import Property, QObject, QRect, QUrl, Signal, Slot
from PySide6.QtGui import QImage
from PySide6.QtQml import QQmlApplicationEngine, QQmlComponent

from assistant.facade import vii
from assistant.snippet_actions import hide_snippet_overlays, show_snippet_overlays

log = logging.getLogger(__name__)

QML_DIR = Path(__file__).parent.parent / "qml"


class SnippetViewModel(QObject):
    """QML-facing view model for the snippet overlay.

    Manages lifecycle of overlay windows in response to view state list changes.
    """

    activeChanged = Signal()
    region_captured = Signal(QImage)  # Emitted with the captured region image

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self._active = False
        self._overlay_windows: list[QObject] = []
        self._component = None  # Keep QML component alive to prevent GC
        app_state.snippet_overlays_changed.connect(self.sync_from_app_state)

    def _get_active(self) -> bool:
        return self._active

    active = Property(bool, _get_active, notify=activeChanged)

    @Slot()
    def activate(self):
        """Trigger the show_snippet_overlays action."""
        show_snippet_overlays(vii.app.view_state)

    @Slot()
    def cancel(self):
        """Cancel snippet selection via the action."""
        hide_snippet_overlays(vii.app.view_state)

    @Slot(str, float, float, float, float)
    def region_selected(self, screen_name: str, x: float, y: float, w: float, h: float):
        """Called from QML when the user finishes dragging a selection.

        Crops the region from the screenshot captured before the overlay
        appeared, so the crosshatch overlay never leaks into the snippet.
        """
        x, y, w, h = int(x), int(y), int(w), int(h)

        if not self._active:
            log.warning("region_selected called but not active, ignoring")
            return

        log.info("Region selected on screen '%s': %d,%d %dx%d", screen_name, x, y, w, h)

        rect = QRect(x, y, w, h)
        if not rect.isValid() or rect.width() < 5 or rect.height() < 5:
            log.warning("Selected region too small, ignoring")
            hide_snippet_overlays(vii.app.view_state)
            return

        # Grab the pre-overlay screenshot before hide clears the view states.
        screenshot = self._find_screenshot(screen_name)

        # Hide overlays now that we have the clean screenshot.
        hide_snippet_overlays(vii.app.view_state)

        if screenshot is None or screenshot.isNull():
            log.error("No pre-overlay screenshot for screen '%s'", screen_name)
            return

        self._crop_region(screenshot, rect)

    def sync_from_app_state(self):
        """React to changes in app_state.snippet_overlays.

        Creates or destroys QML overlay windows to match the view state list.
        """
        overlays = vii.app.view_state.snippet_overlays
        new_active = len(overlays) > 0

        log.info(
            "sync_from_app_state: new_active=%s, self._active=%s, windows=%d",
            new_active,
            self._active,
            len(self._overlay_windows),
        )

        if new_active and not self._active:
            # Create overlay windows
            self._create_overlay_windows(overlays)
        elif not new_active and self._active:
            # Destroy overlay windows
            self._destroy_overlay_windows()

        if new_active != self._active:
            self._active = new_active
            self.activeChanged.emit()

    def _create_overlay_windows(self, overlays):
        """Create one QML overlay window per view state."""
        engine: QQmlApplicationEngine = vii.app.engine
        qml_file = QML_DIR / "SnippetOverlay.qml"
        self._component = QQmlComponent(engine, QUrl.fromLocalFile(str(qml_file)))

        if self._component.isError():
            log.error("SnippetOverlay QML errors: %s", self._component.errors())
            return

        for vs in overlays:
            obj = self._component.create(engine.rootContext())
            if obj is None:
                log.error(
                    "Failed to create SnippetOverlay for screen '%s'",
                    vs.screen_name,
                )
                continue
            obj.setProperty("screenName", vs.screen_name)
            geo = vs.geometry
            obj.setProperty("x", geo.x())
            obj.setProperty("y", geo.y())
            obj.setProperty("width", geo.width())
            obj.setProperty("height", geo.height())
            self._overlay_windows.append(obj)
            log.info("SnippetOverlay created for screen '%s'", vs.screen_name)

    def _destroy_overlay_windows(self):
        """Destroy all overlay windows."""
        for i, obj in enumerate(self._overlay_windows):
            valid = shiboken6.isValid(obj)
            log.info("  destroying window %d: valid=%s", i, valid)
            if valid:
                # Allow close by setting the QML property, then delete
                obj.setProperty("visible", False)
                obj.deleteLater()
        self._overlay_windows.clear()
        log.info("All snippet overlays destroyed")

    def _find_screenshot(self, screen_name: str) -> QImage | None:
        """Return the pre-overlay screenshot stored for the given screen."""
        for vs in vii.app.view_state.snippet_overlays:
            if vs.screen_name == screen_name:
                return vs.screenshot
        return None

    def _crop_region(self, screenshot: QImage, rect: QRect):
        """Crop the selected region from the pre-overlay screenshot."""
        # Account for device pixel ratio (HiDPI); rect is in logical coords.
        dpr = screenshot.devicePixelRatio()
        physical_rect = QRect(
            int(rect.x() * dpr),
            int(rect.y() * dpr),
            int(rect.width() * dpr),
            int(rect.height() * dpr),
        )
        region_image = screenshot.copy(physical_rect)

        log.info(
            "Captured snippet: %dx%d pixels",
            region_image.width(),
            region_image.height(),
        )
        self.region_captured.emit(region_image)
