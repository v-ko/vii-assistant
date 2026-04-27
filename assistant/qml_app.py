import sys
from pathlib import Path

from fusion.loop import set_main_loop
from fusion.platform.qt_widgets.qt_main_loop import QtMainLoop
from PySide6.QtCore import QMetaObject, QRect, Qt, QUrl
from PySide6.QtGui import QAction, QScreen
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtWidgets import QApplication, QMenu, QStyle, QSystemTrayIcon

from assistant.app_state import AppState
from assistant.context_list_model import ContextListModel
from assistant.facade import vii
from assistant.qml_backend import QmlBackend
from assistant.terminal_actions import toggle_terminal
from assistant.util import get_logger, get_screen_by_name
from assistant.view_states.terminal import TerminalViewState
from assistant.widgets.overlay import ModelVisionOverlay

log = get_logger(__name__)

QML_DIR = Path(__file__).parent / "qml"


class AssistantQmlApp(QApplication):

    def __init__(self):
        super().__init__(sys.argv)
        set_main_loop(QtMainLoop(self))
        self.setQuitOnLastWindowClosed(False)

        # Initialize view states
        self.app_state = AppState(parent=self)
        self.terminal_state = TerminalViewState(self.app_state)

        # QML backend bridge
        self.qml_backend = QmlBackend(parent=self)

        # Context list model for QML
        self.context_model = ContextListModel(self.app_state.context_VS, parent=self)

        # QML engine
        self.engine = QQmlApplicationEngine()
        ctx = self.engine.rootContext()
        ctx.setContextProperty("settingsState", self.app_state.settings_VS)
        ctx.setContextProperty("contextState", self.app_state.context_VS)
        ctx.setContextProperty("contextModel", self.context_model)
        ctx.setContextProperty("terminalState", self.terminal_state)
        ctx.setContextProperty("backend", self.qml_backend)

        # Add QML import path for local components
        self.engine.addImportPath(str(QML_DIR))

        # Load main QML
        qml_file = QML_DIR / "TerminalWindow.qml"
        self.engine.load(QUrl.fromLocalFile(str(qml_file)))
        if not self.engine.rootObjects():
            log.error("Failed to load QML file: %s", qml_file)
            sys.exit(1)

        self._root_window = self.engine.rootObjects()[0]

        # Overlay will be created after screen is configured
        self.overlay: ModelVisionOverlay | None = None

        self.setup_tray_icon()

    @property
    def terminal_window(self):
        """Compatibility: return the QML root window.

        The QML window exposes showTerminal() / hideTerminal() JS functions
        which can be invoked via QMetaObject.
        """
        return self._root_window

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
        self.qml_backend.screen_list_changed.emit()

    def _on_screen_removed(self, screen: QScreen) -> None:
        log.info(f"Screen removed: {screen.name()}")
        settings = self.terminal_state.app_state.settings_VS
        if screen.name() == settings.screen:
            fallback = vii.get_default_screen()
            log.info(f"Watched screen removed, falling back to: {fallback.name()}")
            settings.screen = fallback.name()
        self.qml_backend.screen_list_changed.emit()

    def _on_primary_screen_changed(self, screen: QScreen) -> None:
        log.info(f"Primary screen changed: {screen.name()}")

    def show_terminal(self):
        """Show the terminal with drop-down animation."""
        if self._root_window:
            QMetaObject.invokeMethod(  # type: ignore[call-overload]
                self._root_window,
                "showTerminal",
                Qt.ConnectionType.QueuedConnection,
            )

    def hide_terminal(self):
        """Hide the terminal with slide-up animation."""
        if self._root_window:
            QMetaObject.invokeMethod(  # type: ignore[call-overload]
                self._root_window,
                "hideTerminal",
                Qt.ConnectionType.QueuedConnection,
            )
