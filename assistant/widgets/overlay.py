from typing import List
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPolygon

from assistant.util import Shape


class ModelVisionOverlay(QWidget):
    """
    A transparent overlay widget that displays shapes on top of the screen.

    The widget is transparent to user events (click-through) and displays
    shapes based on the configuration provided through setShapes.

    Currently supports 'rect' and 'point' shape types.
    """

    def __init__(self, parent=None):
        super().__init__(
            parent,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.X11BypassWindowManagerHint,
        )  # Bypass window manager on X11

        # Make the widget transparent
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        # Make the widget transparent to user events (click-through)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        # List of shapes to draw
        self._shapes: List[Shape] = []

        # Hardcoded style settings (but color comes from shape)
        self.default_color = "#FF0000"  # Red default
        self.shape_width = 2
        self.triangle_size = 40  # Size of triangle cursor for points

        # Set up the widget to cover the entire screen
        self._setup_full_screen()

    def _setup_full_screen(self):
        """Set up the widget to cover the entire screen."""
        screen = self.screen()
        if screen:
            # Use the full screen geometry, not just the available area
            geometry = screen.geometry()
            print(f"Setting overlay geometry to: {geometry}")
            self.setGeometry(geometry)

            # Reset window flags to ensure proper behavior
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.X11BypassWindowManagerHint
            )

            # Make the widget transparent to user events (click-through)
            # This is critical for allowing clicks to pass through
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

            # Show in full screen mode
            self.showFullScreen()

            # Ensure the attribute is still set after showing
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

            print("Overlay should now be transparent for mouse events")

    def setShapes(self, shapes: List[Shape]):
        """
        Set the shapes to be drawn on the overlay.

        Args:
            shapes: A list of dictionaries with shape properties.
                   Each dictionary should have at least 'type' and 'geometry' keys.
                   Supported types: 'rect', 'point'
        """
        self._shapes = shapes
        self.update()  # Trigger a repaint

    def paintEvent(self, event):
        """Paint the overlay with the configured shapes."""
        painter = QPainter(self)

        # Enable antialiasing for smoother shapes
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw a thin red outline around the widget
        outline_pen = QPen(QColor(255, 0, 0))  # Red
        outline_pen.setWidth(10)
        painter.setPen(outline_pen)
        painter.drawRect(self.rect())

        # Draw each shape
        for shape in self._shapes:
            shape_type = shape.get("type", "")
            geometry = shape.get("geometry", None)

            if not geometry:
                print("Invalid shape geometry, skipping:", shape)
                continue

            # Set up pen with color from shape (or default) and hardcoded width
            pen = QPen(QColor(shape.get("color", self.default_color)))
            pen.setWidth(self.shape_width)
            painter.setPen(pen)

            # Set up brush for filled shapes
            brush = QBrush(QColor(shape.get("color", self.default_color)))
            painter.setBrush(brush)

            # Draw the shape based on its type
            if shape_type == "rect":
                if isinstance(geometry, (list, tuple)) and len(geometry) == 4:
                    x, y, width, height = geometry
                    painter.drawRect(x, y, width, height)

            elif shape_type == "point":
                if isinstance(geometry, (list, tuple)) and len(geometry) >= 2:
                    x, y = geometry[0], geometry[1]
                    print("Drawing point at:", x, y)
                    print("In self rect:", self.rect())

                    # Draw a mouse cursor-like triangle pointing to the point
                    # Make it inclined and sharp (less than 60 degrees)
                    size = self.triangle_size

                    # Create an inclined triangle pointing bottom-right
                    triangle = QPolygon([
                        QPoint(x, y),  # The exact point (tip of cursor)
                        QPoint(x + size//3, y + size),  # Bottom right (narrow)
                        QPoint(x + size//2, y + size//2),  # Middle right (wider)
                        QPoint(x + size, y + size//3)   # Top right
                    ])
                    painter.drawPolygon(triangle)
