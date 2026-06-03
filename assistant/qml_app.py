import sys
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QUrl
from PySide6.QtGui import QAction, QScreen
from PySide6.QtQml import QQmlApplicationEngine, QQmlComponent
from PySide6.QtWidgets import QApplication, QMenu, QStyle, QSystemTrayIcon
from sivkit.loop import set_main_loop
from sivkit.platform.qt_widgets.qt_main_loop import QtMainLoop

from assistant.app_state import AppViewState
from assistant.app_view_model import AppViewModel
from assistant.context_list_model import ContextListModel
from assistant.facade import vii
from assistant.projections import project_screen_layout
from assistant.services.recording_overlay_view_model import RecordingOverlayViewModel
from assistant.services.settings_modal_view_model import SettingsModalViewModel
from assistant.services.snippet_view_model import SnippetViewModel
from assistant.terminal_actions import toggle_terminal
from assistant.util import get_logger, get_screen_by_name
from assistant.view_states.screen_info import ScreenInfoData
from assistant.view_states.terminal import TerminalViewState
from assistant.widgets.overlay import ModelVisionOverlay

log = get_logger(__name__)

QML_DIR = Path(__file__).parent / "qml"


class ViiQmlApp(QApplication):

    def __init__(self, view_state: AppViewState):
        super().__init__(sys.argv)
        set_main_loop(QtMainLoop(self))
        self.setQuitOnLastWindowClosed(False)

        self.view_state = view_state
        self.terminal_state = TerminalViewState(self.view_state)

        # View models (only need app_state, no QML engine)
        self.app_view_model = AppViewModel(parent=self)
        self.context_model = ContextListModel(self.view_state.context_VS, parent=self)
        self.recording_overlay_view_model = RecordingOverlayViewModel(view_state)
        self.settings_modal_view_model = SettingsModalViewModel(view_state)
        self.snippet_view_model = SnippetViewModel(view_state)

        # Will be initialized in init_ui()
        self._engine: QQmlApplicationEngine | None = None
        self._root_window = None
        self._recording_overlay_component = None
        self._recording_overlay_windows: list[QObject] = []
        self.overlay: ModelVisionOverlay | None = None
        self._screen_debug_widget = None

    def init_ui(self) -> None:
        """Create QML engine, load UI, set up tray icon.

        Must be called after all services are registered on the facade.
        """
        # QML engine
        self._engine = QQmlApplicationEngine()
        ctx = self.engine.rootContext()
        ctx.setContextProperty("settingsState", self.view_state.settings_VS)
        ctx.setContextProperty("contextState", self.view_state.context_VS)
        ctx.setContextProperty("contextModel", self.context_model)
        ctx.setContextProperty("terminalState", self.terminal_state)
        ctx.setContextProperty("appVM", self.app_view_model)
        ctx.setContextProperty("recordingOverlayVM", self.recording_overlay_view_model)
        ctx.setContextProperty("settingsModalVM", self.settings_modal_view_model)
        ctx.setContextProperty("snippetVM", self.snippet_view_model)

        # Add QML import path for local components
        self.engine.addImportPath(str(QML_DIR))

        # Load main QML
        qml_file = QML_DIR / "TerminalWindow.qml"
        self.engine.load(QUrl.fromLocalFile(str(qml_file)))
        if not self.engine.rootObjects():
            log.error("Failed to load QML file: %s", qml_file)
            sys.exit(1)

        self._root_window = self.engine.rootObjects()[0]

        # Load recording overlay QML (one window per screen)
        self._create_recording_overlays()

        # Screen debug widget (lazy, shown via view state signal)
        self.view_state.screen_debug_visible_changed.connect(
            self._on_screen_debug_changed
        )

        self.setup_tray_icon()

    @property
    def engine(self) -> QQmlApplicationEngine:
        if self._engine is None:
            raise RuntimeError("QML engine not initialized yet. Call init_ui() first.")
        return self._engine

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
            self.overlay = ModelVisionOverlay(self.view_state.overlay_VS)
            self.overlay.hide()

        if getattr(self, "_overlay_bound", False):
            return
        self._overlay_bound = True

        self.screenAdded.connect(self._on_screen_config_changed)
        self.screenRemoved.connect(self._on_screen_config_changed)
        self.primaryScreenChanged.connect(self._on_screen_config_changed)

        screen = get_screen_by_name(initial_screen_name)
        if screen:
            self._watch_screen_geometry(screen)

        # React to capture_screen config changes (via projector → VS signal)
        self.view_state.settings_VS.capture_screen_changed.connect(
            self._on_capture_screen_changed, Qt.ConnectionType.QueuedConnection
        )

        # Bind view model to screen changes, then run initial projection
        self.app_view_model.bind_screens()
        self._project_screens()

        # Overlay follows terminal visibility
        self.terminal_state.visible_changed.connect(self._on_terminal_visible_changed)

    def _compile_screen_info(self) -> list[ScreenInfoData]:
        """Serialize current Qt screen state into plain data for the projector."""
        primary = self.primaryScreen()
        capture_name = vii.get_config().capture_screen

        # Validate capture screen — fallback to default if gone
        capture_exists = any(s.name() == capture_name for s in self.screens())
        if not capture_exists:
            fallback = vii.get_default_screen()
            capture_name = fallback.name()
            cfg = vii.get_config()
            cfg.capture_screen = capture_name
            vii.update_config(cfg)

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
        project_screen_layout(self._compile_screen_info(), self.view_state)

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
        self._create_recording_overlays()
        self.app_view_model.screen_list_changed.emit()

    def _on_capture_screen_changed(self, screen_name: str) -> None:
        """React to capture_screen config change (via projector → VS signal)."""
        screen = get_screen_by_name(screen_name)
        if screen:
            self._watch_screen_geometry(screen)
        self._project_screens()

    def _on_terminal_visible_changed(self, visible: bool) -> None:
        """Sync overlay visibility with terminal visibility."""
        if self.overlay:
            if visible:
                self.overlay.show()
            else:
                self.overlay.hide()

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

    # ── Recording overlay (one per screen) ───────────────────────

    def _create_recording_overlays(self) -> None:
        """Create one RecordingOverlay QML window per screen."""
        self._destroy_recording_overlays()

        pill_qml = QML_DIR / "RecordingOverlay.qml"
        if self._recording_overlay_component is None:
            self._recording_overlay_component = QQmlComponent(
                self.engine, QUrl.fromLocalFile(str(pill_qml))
            )
        if self._recording_overlay_component.isError():
            log.error(
                "RecordingOverlay QML errors: %s",
                self._recording_overlay_component.errors(),
            )
            return

        screen = self.primaryScreen()
        if screen is None:
            log.error("No primary screen available for RecordingOverlay")
            return
        obj = self._recording_overlay_component.create(self.engine.rootContext())
        if obj is None:
            log.error(
                "Failed to create RecordingOverlay for screen '%s'",
                screen.name(),
            )
            return
        geo = screen.geometry()
        obj.setProperty("screenX", geo.x())
        obj.setProperty("screenY", geo.y())
        obj.setProperty("screenWidth", geo.width())
        obj.setProperty("screenHeight", geo.height())
        self._recording_overlay_windows.append(obj)

    def _destroy_recording_overlays(self) -> None:
        """Destroy all recording overlay windows."""
        for obj in self._recording_overlay_windows:
            obj.setProperty("visible", False)
            obj.deleteLater()
        self._recording_overlay_windows.clear()
