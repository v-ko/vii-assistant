import sys

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QMenu, QStyle, QSystemTrayIcon

from assistant.app_actions import toggle_terminal
from assistant.util import get_logger
from assistant.view_states.terminal import TerminalViewState
from assistant.widgets.overlay import ModelVisionOverlay
from assistant.widgets.terminal import TerminalWindow

log = get_logger(__name__)


class AssistantQtApp(QApplication):

    def __init__(self):
        super().__init__(sys.argv)
        # Prevent app from closing when all windows are closed
        self.setQuitOnLastWindowClosed(False)

        # Initialize components
        self.terminal_state = TerminalViewState()
        self.terminal_window = TerminalWindow(self.terminal_state)

        # Create the overlay and show it
        # The screen will be set properly in facade.py when apply_config is called
        self.overlay = ModelVisionOverlay()
        # Show the overlay immediately
        self.overlay.show()

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
