import sys

from fusion.loop import set_main_loop
from fusion.platform.qt_widgets.qt_main_loop import QtMainLoop
from PySide6.QtCore import QRect
from PySide6.QtGui import QAction, QScreen
from PySide6.QtWidgets import QApplication, QMenu, QStyle, QSystemTrayIcon

from assistant.app_actions import toggle_terminal
from assistant.app_state import AppState
from assistant.facade import facade
from assistant.util import get_logger, get_screen_by_name
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
        self.overlay: ModelVisionOverlay | None = None

        self.setup_tray_icon()

    def setup_tray_icon(self):
        self.tray_icon = QSystemTrayIcon()
        self.tray_icon.setToolTip("Desktop Screenshot Assistant")
        self.tray_icon.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
        )

        tray_menu = QMenu()

        self.terminal_action = QAction("Toggle Terminal", self)
        self.terminal_action.triggered.connect(toggle_terminal)
        tray_menu.addAction(self.terminal_action)

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.quit)
        tray_menu.addAction(quit_action)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

    def bind_screen_overlay(self, initial_screen_name: str):
        if self.overlay is None:
            self.overlay = ModelVisionOverlay(self.app_state.overlay_VS)
            self.overlay.hide()

        self.app_state.overlay_VS.screen_name = initial_screen_name

        if getattr(self, "_overlay_bound", False):
            return
        self._overlay_bound = True

        settings_state = self.terminal_state.app_state.settings_VS
        settings_state.screen_changed.connect(self._on_screen_changed)

        self.screenAdded.connect(self._on_screen_added)
        self.screenRemoved.connect(self._on_screen_removed)
        self.primaryScreenChanged.connect(self._on_primary_screen_changed)

        screen = get_screen_by_name(initial_screen_name)
        if screen:
            self._watch_screen_geometry(screen)

    def _watch_screen_geometry(self, screen: QScreen) -> None:
        prev = getattr(self, "_watched_screen", None)
        if prev is not None:
            try:
                prev.geometryChanged.disconnect(self._on_screen_geometry_changed)
            except RuntimeError:
                pass
        self._watched_screen = screen
        screen.geometryChanged.connect(self._on_screen_geometry_changed)

    def _on_screen_geometry_changed(self, rect: QRect) -> None:
        log.info(f"Screen geometry changed: {rect}")
        # Re-set screen_name to trigger overlay repositioning
        overlay_vs = self.app_state.overlay_VS
        name = overlay_vs.screen_name
        overlay_vs._screen_name = ""
        overlay_vs.screen_name = name

    def _on_screen_changed(self, screen_name: str) -> None:
        if not screen_name:
            return
        self.app_state.overlay_VS.screen_name = screen_name
        screen = get_screen_by_name(screen_name)
        if screen:
            self._watch_screen_geometry(screen)

    def _on_screen_added(self, screen: QScreen) -> None:
        log.info(f"Screen added: {screen.name()}")
        self._repopulate_screen_combo()
        self.terminal_window.position_window()

    def _on_screen_removed(self, screen: QScreen) -> None:
        log.info(f"Screen removed: {screen.name()}")
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

    def _repopulate_screen_combo(self) -> None:
        try:
            sw = self.terminal_window.model_settings
            sw.populate_screen_combo()
        except Exception:
            pass
