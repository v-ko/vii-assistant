import sys
from pathlib import Path

from fusion.loop import set_main_loop
from fusion.platform.qt_widgets.qt_main_loop import QtMainLoop
from PySide6.QtCore import QMetaObject, Qt, QUrl
from PySide6.QtGui import QAction, QScreen
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtWidgets import QApplication, QMenu, QStyle, QSystemTrayIcon

from assistant.app_state import AppState
from assistant.app_view_model import AppViewModel
from assistant.context_list_model import ContextListModel
from assistant.facade import vii
from assistant.projections import project_screen_layout
from assistant.terminal_actions import toggle_terminal
from assistant.util import get_logger, get_screen_by_name
from assistant.view_states.screen_info import ScreenInfoData
from assistant.view_states.terminal import TerminalViewState
from assistant.widgets.overlay import ModelVisionOverlay

log = get_logger(__name__)

QML_DIR = Path(__file__).parent / "qml"


class AssistantQmlApp(QApplication):

    def __init__(self, app_state: AppState):
        super().__init__(sys.argv)
        set_main_loop(QtMainLoop(self))
        self.setQuitOnLastWindowClosed(False)

        self.app_state = app_state
        self.terminal_state = TerminalViewState(self.app_state)

        # App-level QML ViewModel
        self.app_view_model = AppViewModel(parent=self)

        # Context list model for QML
        self.context_model = ContextListModel(self.app_state.context_VS, parent=self)

        # QML engine
        self.engine = QQmlApplicationEngine()
        ctx = self.engine.rootContext()
        ctx.setContextProperty("settingsState", self.app_state.settings_VS)
        ctx.setContextProperty("contextState", self.app_state.context_VS)
        ctx.setContextProperty("contextModel", self.context_model)
        ctx.setContextProperty("terminalState", self.terminal_state)
        ctx.setContextProperty("appVM", self.app_view_model)
        ctx.setContextProperty("recordingOverlayVM", vii.recording_overlay_view_model)
        ctx.setContextProperty("settingsModalVM", vii.settings_modal_view_model)
        ctx.setContextProperty("snippetVM", vii.snippet_view_model)

        # Add QML import path for local components
        self.engine.addImportPath(str(QML_DIR))

        # Load main QML
        qml_file = QML_DIR / "TerminalWindow.qml"
        self.engine.load(QUrl.fromLocalFile(str(qml_file)))
        if not self.engine.rootObjects():
            log.error("Failed to load QML file: %s", qml_file)
            sys.exit(1)

        self._root_window = self.engine.rootObjects()[0]

        # Load recording overlay QML (separate window)
        pill_qml = QML_DIR / "RecordingOverlay.qml"
        self.engine.load(QUrl.fromLocalFile(str(pill_qml)))
        self._recording_overlay_window = self.engine.rootObjects()[-1]

        # Overlay will be created after screen is configured
        self.overlay: ModelVisionOverlay | None = None

        # Screen debug widget (lazy, shown via view state signal)
        self._screen_debug_widget = None
        app_state.screen_debug_visible_changed.connect(self._on_screen_debug_changed)

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

        if getattr(self, "_overlay_bound", False):
            return
        self._overlay_bound = True

        settings_state = self.terminal_state.app_state.settings_VS
        settings_state.screen_changed.connect(self._on_screen_setting_changed)

        self.screenAdded.connect(self._on_screen_config_changed)
        self.screenRemoved.connect(self._on_screen_config_changed)
        self.primaryScreenChanged.connect(self._on_screen_config_changed)

        screen = get_screen_by_name(initial_screen_name)
        if screen:
            self._watch_screen_geometry(screen)

        # Bind view model to screen changes, then run initial projection
        self.app_view_model.bind_screens()
        self._project_screens()

    def _compile_screen_info(self) -> list[ScreenInfoData]:
        """Serialize current Qt screen state into plain data for the projector."""
        primary = self.primaryScreen()
        capture_name = self.app_state.settings_VS.screen

        # Validate capture screen — fallback to default if gone
        capture_exists = any(s.name() == capture_name for s in self.screens())
        if not capture_exists:
            fallback = vii.get_default_screen()
            capture_name = fallback.name()
            self.app_state.settings_VS.screen = capture_name

        result = []
        for s in self.screens():
            result.append(
                ScreenInfoData(
                    name=s.name(),
                    x=s.geometry().x(),
                    y=s.geometry().y(),
                    width=s.geometry().width(),
                    height=s.geometry().height(),
                    is_primary=(s is primary),
                    is_capture=(s.name() == capture_name),
                )
            )
        return result

    def _project_screens(self) -> None:
        """Compile screen info and run the projector."""
        project_screen_layout(self._compile_screen_info())

    def _watch_screen_geometry(self, screen: QScreen) -> None:
        prev = getattr(self, "_watched_screen", None)
        if prev is not None:
            try:
                prev.geometryChanged.disconnect(self._on_screen_config_changed)
            except RuntimeError:
                pass
        self._watched_screen = screen
        screen.geometryChanged.connect(self._on_screen_config_changed)

    def _on_screen_config_changed(self, *_args) -> None:
        """Single handler for all screen configuration changes."""
        self._project_screens()
        self.app_view_model.screen_list_changed.emit()

    def _on_screen_setting_changed(self, screen_name: str) -> None:
        """User changed the watched screen in settings."""
        if not screen_name:
            return
        screen = get_screen_by_name(screen_name)
        if screen:
            self._watch_screen_geometry(screen)
        self._project_screens()

    def _on_screen_debug_changed(self, visible: bool) -> None:
        """Show or hide the screen debug widget based on view state."""
        if visible:
            if self._screen_debug_widget is None:
                from assistant.widgets.screen_debug import ScreenDebugWidget

                self._screen_debug_widget = ScreenDebugWidget()
            self._screen_debug_widget.show()
        else:
            if self._screen_debug_widget is not None:
                self._screen_debug_widget.hide()

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
