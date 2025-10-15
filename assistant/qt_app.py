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

        # Minimal global styling for buttons (adds depth & focus without heavy theming)
        self.setStyleSheet(
            """
            QPushButton {
                background-color: #444;
                border: 1px solid #666;
                border-radius: 4px;
                padding: 4px 10px; /* more hit area */
                color: #f0f0f0;
                font-weight: 500;
                min-height: 26px; /* avoid slim buttons */
            }
            QPushButton:hover {
                background-color: #515151;
                border-color: #7a7a7a;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
                border-color: #555;
                padding-top: 5px;  /* subtle pressed shift */
                padding-bottom: 3px;
            }
            QPushButton:disabled {
                background-color: #2f2f2f;
                color: #777;
                border-color: #444;
            }
            /* Keep combos consistent */
            QComboBox {
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 4px 8px;
                color: #e0e0e0;
            }
            QComboBox:hover { background-color: #444; }
            """
        )

        # Initialize components
        self.app_state = AppState(parent=self)
        self.terminal_state = TerminalViewState(self.app_state)
        self.terminal_window = TerminalWindow(self.terminal_state)

        # Defer overlay full-screen setup until screen selection applied by facade
        self.overlay = ModelVisionOverlay()
        self.overlay.hide()  # hide until screen assigned

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

    # Helper to show overlay after screen configured
    def show_overlay_on_screen(self, screen):
        try:
            if screen is not None:
                self.overlay.setScreen(screen)
            self.overlay._setup_full_screen()
            self.overlay.show()
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to show overlay: {exc}")

    # --- Screen change binding (moved from facade) ------------------
    def bind_screen_overlay(self):
        """Connect settings screen_changed signal to overlay update once."""
        try:
            settings_state = self.terminal_state.app_state.settings_VS
        except Exception:
            return
        # Avoid duplicate connections: Qt doesn't give an easy handle, so we use an attr flag
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
        self.show_overlay_on_screen(screen)
