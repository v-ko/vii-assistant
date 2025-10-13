from __future__ import annotations

from typing import TYPE_CHECKING, Optional
from uuid import uuid4

from fusion.libs.entity.change import Change
from PySide6.QtGui import QGuiApplication, QPixmap, QScreen

from assistant.app_state import AppState
from assistant.config import Config
from assistant.inference.context import ContextItem, ContextManager
from assistant.services.automation import AutomationService
from assistant.services.base_client import BaseClient
from assistant.services.config_persistence_service import ConfigPersistenceService
from assistant.services.ocr import ocr_sync, start_ocr
from assistant.services.ollama_client import OllamaClient
from assistant.services.project_manager import SessionManager, ViiProjectManager
from assistant.services.session_recorder import SessionRecorderConfig
from assistant.utils.capture_utils import clipboard_image

if TYPE_CHECKING:
    from assistant.qt_app import AssistantQtApp


class Facade:
    _qt_app = None
    _desktop_server = None
    _config = None
    _automation_service = None
    _model_client: Optional[BaseClient] = None
    _watched_screen = None
    _project_manager: Optional[ViiProjectManager] = None
    _session_manager: Optional[SessionManager] = None
    _app_state: Optional["AppState"] = None
    _context_manager: Optional[ContextManager] = None
    _config_persistence: Optional[ConfigPersistenceService] = None

    def __init__(self):
        # Setup config
        self._config = Config()

        self._automation_service = AutomationService()
        # Initialize single hardcoded client (Ollama by default)
        self._model_client = OllamaClient()
        # Initialize project storage in the configuration directory by default
        self._project_manager = ViiProjectManager(self._config.config_dir)
        self._session_manager = None
        self._context_manager = ContextManager()

    @property
    def qt_app(self) -> "AssistantQtApp":
        if self._qt_app is None:
            raise RuntimeError("Qt app instance not set")
        return self._qt_app

    @property
    def config(self) -> "Config":
        if self._config is None:
            raise RuntimeError("Config instance not initialized")
        return self._config

    @property
    def automation(self) -> AutomationService:
        if self._automation_service is None:
            raise RuntimeError("Automation service not initialized")
        return self._automation_service

    @property
    def context_manager(self) -> ContextManager:
        if self._context_manager is None:
            self._context_manager = ContextManager()
        return self._context_manager

    @property
    def inference_client(self) -> BaseClient:
        if self._model_client is None:
            raise RuntimeError("Client not initialized")
        return self._model_client

    @property
    def project_manager(self) -> ViiProjectManager:
        if self._project_manager is None:
            raise RuntimeError("Project manager not initialized")
        return self._project_manager

    @property
    def app_state(self) -> AppState:
        if self._app_state is None:
            raise RuntimeError("App state not initialized")
        return self._app_state

    # Model/client selection removed; single client in use.

    @property
    def watched_screen(self) -> Optional[QScreen]:
        """Get the watched screen."""
        return self._watched_screen

    def setQtApp(self, app: "AssistantQtApp"):
        if self._qt_app is not None:
            raise RuntimeError("setQtApp called more than once")
        self._qt_app = app

        terminal_state = self._qt_app.terminal_state
        app_state = terminal_state.app_state
        self._app_state = app_state

        settings_state = app_state.settings
        # Ensure a valid screen value exists in config BEFORE initializing settings state
        try:
            screen_name = self.config.get("screen", "")
            screen_valid = False
            if screen_name:
                screen_valid = self.get_screen_by_name(screen_name) is not None
            if not screen_valid:
                # Pick default screen (second non-primary if available else first)
                try:
                    default_screen = self.get_default_screen()
                    if default_screen:
                        self.config.set("screen", default_screen.name())
                except Exception:
                    pass
        except Exception:
            pass

        settings_state.initialize(self.config, self.project_manager)
        # Bind config persistence AFTER settings initialized so initial apply_config does not trigger writes
        self._config_persistence = ConfigPersistenceService()
        self._config_persistence.bind(self.config, settings_state)
        # React to runtime changes (screen updates only)
        settings_state.screen_changed.connect(self._on_screen_changed)

        settings_state.system_prompt_changed.connect(
            lambda prompt: self.automation.set_system_prompt(prompt)
        )
        self.automation.set_system_prompt(settings_state.system_prompt_markdown)

        terminal_window = self._qt_app.terminal_window
        terminal_window.set_facade(self)
        context_widget = terminal_window.context_viewer
        context_widget.message_submitted.connect(self._handle_message_submitted)

        # UI signal wiring moved to app layer; only manual context submission retained here.

    def start_desktop_server(self, port: int):
        """Start the desktop server."""
        from assistant.server.desktop_server import DesktopServer

        server = DesktopServer(port)
        server.start()
        self._desktop_server = server

    def _on_screen_changed(self, screen_name: str) -> None:
        if not screen_name:
            return
        screen = self.get_screen_by_name(screen_name)
        if not screen:
            try:
                screen = self.get_default_screen()
            except Exception:
                return
        self._watched_screen = screen
        if self._qt_app and self._qt_app.overlay:
            self._qt_app.overlay.setScreen(screen)
            self._qt_app.overlay._setup_full_screen()

    # Session control moved to actions module.

    def add_context_item(self, item: ContextItem) -> None:
        manager = self.context_manager
        manager.insert(item)
        change = Change.CREATE(item)
        self.app_state.context.apply_change(change)

    def _handle_message_submitted(self, text: str) -> None:
        manager = self.context_manager
        cleaned = text.strip()
        if cleaned:
            text_item = ContextItem()
            text_item.position = manager.next_position()
            text_item.size = 0
            text_item.content = {"text": cleaned}
            text_item.metadata = {"origin": "user"}
            self.add_context_item(text_item)

        request_item = ContextItem()
        request_item.position = manager.next_position()
        request_item.size = 0
        request_item.content = {"text": ""}
        request_item.request = {
            "stream": True,
            "max_new_tokens": 256,
            "temperature": 0.0,
        }
        request_item.metadata = {"origin": "user", "trigger": "manual-send"}
        self.add_context_item(request_item)

    def get_screen_by_name(self, name: str) -> Optional[QScreen]:
        """Get a QScreen object by its name."""
        screens = QGuiApplication.screens()
        for screen in screens:
            if screen.name() == name:
                return screen
        return None

    def get_default_screen(self) -> QScreen:
        """Get the default screen (second to primary if available)."""
        screens = QGuiApplication.screens()
        if len(screens) > 1:
            primary = QGuiApplication.primaryScreen()
            screens.remove(primary)
        return screens[0]  # First non-primary screen

    def ocr_clipboard_action(self):
        """OCR clipboard image and copy recognized text to clipboard.

        Still updates terminal output for user visibility.
        Uses local Tesseract via pytesseract (non-blocking worker thread).
        """
        pixmap = clipboard_image()
        if not pixmap:
            msg = "No image found in clipboard for OCR"
            print(msg)
            self.qt_app.terminal_state.output_text = msg
            return

        # Indicate progress
        self.app_state.settings.request_in_progress = True

        self._ocr_worker = start_ocr(
            pixmap,
            on_finished=self._on_ocr_finished_copy,
            on_error=lambda m: self._on_ocr_finished_copy(m),
        )

    def _on_ocr_finished_copy(self, text: str):
        self.app_state.settings.request_in_progress = False
        # Copy result to system clipboard (even if it's an error or <No text...>)
        try:
            cb = QGuiApplication.clipboard()
            cb.setText(text)
        except Exception as e:  # pragma: no cover - clipboard failure rare
            text = f"{text}\n(Clipboard copy failed: {e})"
        self.qt_app.terminal_state.output_text = text

    # --- Getter API for external callers ---
    def get_clipboard_ocr_text(self, lang: str = "eng") -> str:
        """Get OCR text from the current clipboard image synchronously.

        Args:
            lang: Tesseract language code (default "eng").

        Returns:
            Recognized text, or an error message starting with "Error:".
        """
        pixmap = clipboard_image()
        if not pixmap:
            return "Error: No image in clipboard"
        try:
            return ocr_sync(pixmap, lang=lang)
        except Exception as e:  # pragma: no cover
            return f"Error: OCR failed: {e}"

    # Removed trivial clipboard wrapper; use clipboard_image() directly.


facade = Facade()
