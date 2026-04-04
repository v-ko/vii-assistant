import sys

from fusion.loop import set_main_loop
from fusion.platform.qt_widgets.qt_main_loop import QtMainLoop
from PySide6.QtCore import QRect
from PySide6.QtGui import QAction, QScreen
from PySide6.QtWidgets import QApplication, QMenu, QStyle, QSystemTrayIcon

from assistant.app_actions import toggle_terminal
from assistant.app_state import AppState
from assistant.facade import facade
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

        #  Initialize components
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
        self.initialize_overlay(initial_screen)

        try:
            settings_state = self.terminal_state.app_state.settings_VS
        except Exception:
            return
        if getattr(self, "_overlay_bound", False):
            return
        settings_state.screen_changed.connect(self._on_screen_changed)

        self.screenAdded.connect(self._on_screen_added)
        self.screenRemoved.connect(self._on_screen_removed)
        self.primaryScreenChanged.connect(self._on_primary_screen_changed)
        self._watch_screen_geometry(initial_screen)
        self._overlay_bound = True

    def _watch_screen_geometry(self, screen: QScreen) -> None:
        prev = getattr(self, "_watched_screen", None)
        if prev is not None:
            try:
                prev.geometryChanged.disconnect(self._on_screen_geometry_changed)
            except RuntimeError:
                pass
        self._watched_screen = screen
        screen.geometryChanged.connect(self._on_screen_geometry_changed)

    def _on_screen_changed(self, screen_name: str) -> None:
        if not screen_name:
            return
        screen = facade.get_screen_by_name(screen_name)
        if screen is None:
            raise RuntimeError(f"Selected screen '{screen_name}' not found")
        self._watch_screen_geometry(screen)
        self.update_overlay_screen(screen)

    def _on_screen_added(self, screen: QScreen) -> None:
        log.info(f"Screen added: {screen.name()}")
        self._repopulate_screen_combo()
        self.terminal_window.position_window()

    def _on_screen_removed(self, screen: QScreen) -> None:
        log.info(f"Screen removed: {screen.name()}")
        from assistant.facade import facade

        settings = self.terminal_state.app_state.settings_VS
        if screen.name() == settings.screen:
            fallback = facade.get_default_screen()
            log.info(f"Watched screen removed, falling back to: {fallback.name()}")
            settings.screen = fallback.name()
        self._repopulate_screen_combo()
        self.terminal_window.position_window()

    def _on_primary_screen_changed(self, screen: QScreen) -> None:
        log.info(f"Primary screen changed: {screen.name()}")
        self.terminal_window.position_window()

    def _on_screen_geometry_changed(self, rect: QRect) -> None:
        log.info(f"Screen geometry changed: {rect}")
        if self.overlay is not None:
            self.overlay._setup_full_screen()

    def _repopulate_screen_combo(self) -> None:
        try:
            sw = self.terminal_window.model_settings
            sw.populate_screen_combo()
        except Exception:
            pass
