"""Debug window that shows a zoomed-out view of all screens and window positions.

Launch via: from assistant.widgets.screen_debug import ScreenDebugWidget; w = ScreenDebugWidget(); w.show()
Or call show_screen_debug() from qml_app after init.
"""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

SCALE = 0.1  # 10% zoom — 1920px screen becomes 192px in the debug view
MARGIN = 20


class ScreenDebugWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Screen Debug")
        self.setMinimumSize(600, 400)
        self.resize(800, 500)

        # Poll every 500ms
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(500)

    def closeEvent(self, event):
        """Reset view state flag when user closes the window."""
        from assistant.debug_actions import hide_screen_debug

        hide_screen_debug()
        super().closeEvent(event)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(30, 30, 30))

        app = QApplication.instance()
        screens = app.screens()

        # Find bounding box of all screens to center the view
        min_x = min(s.geometry().x() for s in screens)
        min_y = min(s.geometry().y() for s in screens)

        offset_x = MARGIN - int(min_x * SCALE)
        offset_y = MARGIN - int(min_y * SCALE)

        font = QFont("monospace", 9)
        p.setFont(font)

        # Draw screens
        for s in screens:
            geo = s.geometry()
            rx = offset_x + int(geo.x() * SCALE)
            ry = offset_y + int(geo.y() * SCALE)
            rw = int(geo.width() * SCALE)
            rh = int(geo.height() * SCALE)

            p.setPen(QPen(QColor(100, 200, 100), 2))
            p.drawRect(rx, ry, rw, rh)

            label = f"{s.name()}\n{geo.width()}x{geo.height()}\n({geo.x()},{geo.y()})"
            p.setPen(QColor(100, 200, 100))
            p.drawText(rx + 4, ry + 14, label.split("\n")[0])
            p.drawText(rx + 4, ry + 26, label.split("\n")[1])
            p.drawText(rx + 4, ry + 38, label.split("\n")[2])

        # Draw windows from the app
        self._draw_windows(p, offset_x, offset_y)

        p.end()

    def _draw_windows(self, p: QPainter, offset_x: int, offset_y: int):
        """Draw outlines of known windows."""
        app = QApplication.instance()

        # Gather all top-level windows
        windows = app.topLevelWindows()

        colors = {
            "TerminalWindow": QColor(100, 150, 255),
            "RecordingOverlay": QColor(255, 100, 100),
            "ModelVisionOverlay": QColor(255, 200, 50),
            "ScreenDebug": QColor(80, 80, 80),  # dim self
        }
        default_color = QColor(200, 200, 200)

        for win in windows:
            if win is None:
                continue
            # Skip invisible windows (except recording overlay which we always want to see)
            geo = win.geometry()
            if geo.width() == 0 or geo.height() == 0:
                continue

            # Determine color by matching window title/objectName
            title = win.title() or win.objectName() or ""
            color = default_color
            for key, c in colors.items():
                if key.lower() in title.lower():
                    color = c
                    break

            rx = offset_x + int(geo.x() * SCALE)
            ry = offset_y + int(geo.y() * SCALE)
            rw = max(2, int(geo.width() * SCALE))
            rh = max(2, int(geo.height() * SCALE))

            if win.isVisible():
                p.setPen(QPen(color, 2))
            else:
                p.setPen(QPen(color, 1, Qt.PenStyle.DashLine))

            p.drawRect(rx, ry, rw, rh)

            # Label
            vis = "V" if win.isVisible() else "H"
            label = f"{title[:20] or '?'} [{vis}] ({geo.x()},{geo.y()}) {geo.width()}x{geo.height()}"
            p.setPen(color)
            p.drawText(rx + 2, ry - 3, label)
