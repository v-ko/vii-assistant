import sys

from fusion.loop import set_main_loop
from fusion.platform.qt_widgets.qt_main_loop import QtMainLoop
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QMenu, QStyle, QSystemTrayIcon

from assistant.app_actions import toggle_terminal
from assistant.app_state import AppState
from assistant.util import get_logger
from assistant.view_states.terminal import TerminalViewState
from assistant.widgets.overlay import ModelVisionOverlay
from assistant.widgets.terminal import TerminalWindow

log = get_logger(__name__)


class AssistantQtApp(QApplication):

    def __init__(self):
        super().__init__(sys.argv)
        set_main_loop(QtMainLoop(self))
        # Prevent app from closing when all windows are closed
        self.setQuitOnLastWindowClosed(False)

        # Global application stylesheet using CSS inheritance principles
        # Using wildcard for universal application of disabled states
        self.setStyleSheet(
            """
            QWidget:disabled {
                background-color: #1a1a1a;
                color: #555;
                border-color: #333;
            }

            QPushButton {
                border: 1px solid #666;
                border-radius: 4px;
                padding: 4px 10px;
                font-weight: 500;
                min-height: 26px;
            }
            QPushButton:enabled {
                background-color: #444;
                color: #f0f0f0;
            }
            QPushButton:enabled:hover {
                background-color: #515151;
                border-color: #7a7a7a;
            }
            QPushButton:enabled:pressed {
                background-color: #3a3a3a;
                border-color: #555;
                padding-top: 5px;
                padding-bottom: 3px;
            }

            QComboBox {
                border: 1px solid #555;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QComboBox:enabled {
                background-color: #3a3a3a;
                color: #e0e0e0;
            }
            QComboBox:enabled:hover {
                background-color: #444;
            }

            QPlainTextEdit {
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px;
                font-family: 'Fira Code', monospace;
            }
            QPlainTextEdit:enabled {
                background-color: #1f1f1f;
                color: #f0f0f0;
            }
        """
        )  # Initialize components
        self.app_state = AppState(parent=self)
        self.terminal_state = TerminalViewState(self.app_state)
        self.terminal_window = TerminalWindow(self.terminal_state)

        # Overlay will be created after screen is configured
        self.overlay = None

        # Set up the tray icon
        self.setup_tray_icon()

    def setup_tray_icon(self):
        # Create the tray icon
        self.tray_icon = QSystemTrayIcon()
        self.tray_icon.setToolTip("Desktop Screenshot Assistant")

        # Use a default icon (you can replace this with a custom icon)
        self.tray_icon.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
        )

        # Create the tray menu
        tray_menu = QMenu()

        # Terminal action
        self.terminal_action = QAction("Toggle Terminal", self)
        self.terminal_action.triggered.connect(toggle_terminal)
        tray_menu.addAction(self.terminal_action)

        # Quit action
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.quit)
        tray_menu.addAction(quit_action)

        # Set the tray menu
        self.tray_icon.setContextMenu(tray_menu)

        # Show the tray icon
        self.tray_icon.show()

    def initialize_overlay(self, screen):
        """Initialize overlay with the configured screen."""
        if self.overlay is not None:
            log.warning("Overlay already initialized")
            return
        log.info(f"Initializing overlay on screen: {screen.name()}")
        self.overlay = ModelVisionOverlay(screen)
        self.overlay.hide()

    def update_overlay_screen(self, screen):
        """Update overlay to a different screen."""
        if self.overlay is None:
            log.error("Cannot update overlay screen - overlay not initialized")
            return
        log.info(f"Updating overlay to screen: {screen.name()}")
        self.overlay.setScreen(screen)
        self.overlay._setup_full_screen()
        # Preserve visibility state
        if self.overlay.isVisible():
            self.overlay.show()

    # --- Screen change binding (moved from facade) ------------------
    def bind_screen_overlay(self, initial_screen):
        """Initialize overlay with screen and connect to screen changes."""
        # Initialize overlay with the configured screen
        self.initialize_overlay(initial_screen)

        # Connect to future screen changes
        try:
            settings_state = self.terminal_state.app_state.settings_VS
        except Exception:
            return
        # Avoid duplicate connections
        if getattr(self, "_overlay_bound", False):
            return
        settings_state.screen_changed.connect(self._on_screen_changed)
        self._overlay_bound = True

    def _on_screen_changed(self, screen_name: str) -> None:
        if not screen_name:
            return
        from assistant.facade import facade  # local import to avoid circular

        screen = facade.get_screen_by_name(screen_name)
        if screen is None:
            raise RuntimeError(f"Selected screen '{screen_name}' not found")
        self.update_overlay_screen(screen)
