from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPolygon,
    QScreen,
)
from PySide6.QtWidgets import QWidget

from assistant.facade import vii
from assistant.util import get_logger, get_screen_by_name
from assistant.view_states.overlay import OverlayMode, OverlayViewState

log = get_logger(__name__)


class ModelVisionOverlay(QWidget):

    def __init__(self, state: OverlayViewState, parent=None):
        super().__init__(
            parent,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.X11BypassWindowManagerHint
            | Qt.WindowType.WindowTransparentForInput,
        )

        self._state = state

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self.default_color = QColor(255, 0, 0, 50)
        self.shape_width = 2
        self.triangle_size = 40

        self._bind_state()

        # Apply initial screen from app_state

        capture = vii.app.view_state.capture_screen_info
        if capture:
            self._apply_screen(capture.name)
        # Listen for screen layout changes
        vii.app.view_state.screens_changed.connect(self._on_screens_changed)

    def _on_screens_changed(self) -> None:
        capture = vii.app.view_state.capture_screen_info
        if capture:
            self._apply_screen(capture.name)

    def _bind_state(self):
        self._state.mode_changed.connect(lambda _: self.update())
        self._state.shapes_changed.connect(lambda _: self.update())
        self._state.gt_shapes_changed.connect(lambda _: self.update())
        self._state.sample_image_changed.connect(lambda _: self.update())
        self._state.pending_actions_changed.connect(lambda _: self.update())

    def _apply_screen(self, screen_name: str):
        if not screen_name:
            return
        screen = get_screen_by_name(screen_name)
        if screen is None:
            log.error(f"Screen '{screen_name}' not found")
            return
        self.setScreen(screen)
        self._setup_full_screen()

    def _setup_full_screen(self):
        """Set up the widget to cover the entire screen."""
        screen = self.screen()
        geometry = screen.geometry()
        log.info(f"Setting overlay geometry to: {geometry} on screen: {screen.name()}")
        self.setGeometry(geometry)

    def show(self):
        """Override to always show fullscreen."""
        self.showFullScreen()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        mode = self._state.mode

        # --- AUTO_GUARD: green tint + stop label ---
        if mode == OverlayMode.AUTO_GUARD:
            painter.fillRect(self.rect(), QColor(0, 200, 0, 25))
            self._draw_centered_label(
                painter, "Alt+F10 to stop assistant", QColor(255, 255, 255, 200)
            )
            return

        # --- CONFIRM: show pending actions for approval ---
        if mode == OverlayMode.CONFIRM:
            painter.fillRect(self.rect(), QColor(0, 0, 0, 40))
            self._draw_confirm_prompt(painter)
            return

        # --- EXPERIMENT: sample image background ---
        if mode == OverlayMode.EXPERIMENT and self._state.sample_image is not None:
            painter.drawImage(
                QRect(0, 0, self.width(), self.height()), self._state.sample_image
            )

        # --- WORK (default): red outline + shapes ---
        # Draw a thin red outline around the widget
        outline_pen = QPen(QColor(255, 0, 0, 50))  # Red
        outline_pen.setWidth(10)
        painter.setPen(outline_pen)
        painter.drawRect(self.rect())

        # Draw each shape (predictions)
        self._draw_shapes(painter, self._state.shapes)

        # Draw ground truth shapes
        self._draw_shapes(painter, self._state.gt_shapes)

    def _draw_centered_label(self, painter: QPainter, text: str, color: QColor) -> None:
        """Draw a large centered label with a subtle shadow."""
        font = QFont("Sans", 28, QFont.Weight.Bold)
        painter.setFont(font)
        rect = self.rect()
        # Shadow
        painter.setPen(QColor(0, 0, 0, 120))
        shadow_rect = QRect(rect.x() + 2, rect.y() + 2, rect.width(), rect.height())
        painter.drawText(shadow_rect, Qt.AlignmentFlag.AlignCenter, text)
        # Text
        painter.setPen(color)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def _draw_confirm_prompt(self, painter: QPainter) -> None:
        """Draw pending actions list and confirmation hint."""
        actions = self._state.pending_actions
        lines = ["Pending actions:"] + [f"  • {a}" for a in actions]
        lines.append("")
        lines.append("Alt+C to confirm  |  Alt+X to cancel")
        text = "\n".join(lines)

        font = QFont("Sans", 18)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255, 230))
        margin = 60
        text_rect = self.rect().adjusted(margin, margin, -margin, -margin)
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
            text,
        )

    def _draw_shapes(self, painter: QPainter, shapes: list):
        """Draw a list of shapes on the painter."""
        for idx, shape in enumerate(shapes):
            shape_type = shape.get("type", "")
            geometry = shape.get("geometry", None)

            if not geometry:
                log.warning(f"Invalid shape geometry, skipping: {shape}")
                continue

            # Set up pen with color from shape (or default) and hardcoded width
            pen = QPen(QColor(shape.get("color", self.default_color)))
            pen.setWidth(self.shape_width)
            painter.setPen(pen)

            # Draw the shape based on its type
            if shape_type == "rect":
                if isinstance(geometry, (list, tuple)) and len(geometry) == 4:
                    x, y, width, height = geometry
                    # Set up filled brush for the rectangle with transparency
                    fill_color = QColor(shape.get("color", self.default_color))
                    fill_color.setAlpha(50)  # Make fill transparent
                    brush = QBrush(fill_color)
                    painter.setBrush(brush)
                    painter.drawRect(x, y, width, height)
                else:
                    log.error(
                        f"Invalid rect geometry format for shape {idx}: {geometry}"
                    )

            elif shape_type == "point":
                if isinstance(geometry, (list, tuple)) and len(geometry) >= 2:
                    x, y = geometry[0], geometry[1]
                    log.debug(f"Drawing point at: {x}, {y}")

                    # Draw a mouse cursor-like triangle pointing to the point
                    # Make it inclined and sharp (less than 60 degrees)
                    size = self.triangle_size

                    # Set up filled brush for the point triangle with transparency
                    fill_color = QColor(shape.get("color", self.default_color))
                    fill_color.setAlpha(50)  # Make fill transparent
                    brush = QBrush(fill_color)
                    painter.setBrush(brush)

                    # Create an inclined triangle pointing bottom-right
                    triangle = QPolygon(
                        [
                            QPoint(x, y),  # The exact point (tip of cursor)
                            QPoint(x + size // 3, y + size),  # Bottom right (narrow)
                            QPoint(
                                x + size // 2, y + size // 2
                            ),  # Middle right (wider)
                            QPoint(x + size, y + size // 3),  # Top right
                        ]
                    )
                    painter.drawPolygon(triangle)
