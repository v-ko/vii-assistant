from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from fusion.libs.channel import Channel
from fusion.libs.entity.change import Change
from PySide6.QtGui import QGuiApplication, QScreen

from assistant.app_state import AppState
from assistant.config import Config
from assistant.inference.context import ContextManager
from assistant.services.config_persistence_service import ConfigPersistenceService

if TYPE_CHECKING:
    from assistant.qt_app import AssistantQtApp
    from assistant.services.project_manager import SessionManager, ViiProjectManager

from fusion import get_logger

log = get_logger(__name__)


class Facade:
    _qt_app = None
    _config = None
    _project_manager: Optional["ViiProjectManager"] = None
    _session_manager: Optional["SessionManager"] = None
    _app_state: Optional["AppState"] = None
    _config_persistence: Optional[ConfigPersistenceService] = None
    _image_preprocessor = None
    _image_preprocessor_model_id: str | None = None

    def __init__(self):
        # Minimal setup; external services injected from main to avoid circular deps
        self._config = Config()
        self._session_manager = None
        # Channels for context sync
        self.client_updates = Channel("client-updates")
        self.inference_updates = Channel("inference-updates")
        self.context_controller = ContextController(self)

    # --- explicit service setters (must be called early in main) ---------
    def set_project_manager(self, manager: ViiProjectManager):
        self._project_manager = manager
        # Wire inference updates to reducer
        self.inference_updates.subscribe(apply_inference_event)

    def set_image_preprocessor_config(self, model_id: str) -> None:
        self._image_preprocessor_model_id = model_id
        self._image_preprocessor = None

    @property
    def image_preprocessor(self):
        if self._image_preprocessor is None:
            if self._image_preprocessor_model_id is None:
                raise RuntimeError("Image preprocessor model not configured")
            from transformers import AutoProcessor

            self._image_preprocessor = AutoProcessor.from_pretrained(
                self._image_preprocessor_model_id
            )
        return self._image_preprocessor

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
    def context_manager(self) -> ContextManager:
        if self._project_manager is None:
            raise RuntimeError("Project manager not set")
        return self._project_manager.context_manager

    @property
    def project_manager(self) -> ViiProjectManager:
        if self._project_manager is None:
            raise RuntimeError(
                "Project manager not set; call setProjectManager in main"
            )
        return self._project_manager

    @property
    def app_state(self) -> AppState:
        if self._app_state is None:
            raise RuntimeError("App state not initialized")
        return self._app_state

    # Model/client selection removed; single client in use.

    def current_watched_screen(self) -> Optional[QScreen]:
        """Return the currently selected screen or raise if unavailable.

        No silent defaulting; caller must ensure a valid selection exists.
        """
        screen_name = self.app_state.settings_VS.screen
        if not screen_name:
            raise RuntimeError("No screen selected in settings")
        scr = self.get_screen_by_name(screen_name)
        if scr is None:
            raise RuntimeError(f"Configured screen '{screen_name}' not found")
        return scr

    def set_qt_app(self, app: "AssistantQtApp"):
        if self._qt_app is not None:
            raise RuntimeError("set_qt_app called more than once")
        self._qt_app = app

        terminal_state = self._qt_app.terminal_state
        app_state = terminal_state.app_state
        self._app_state = app_state

        settings_state = app_state.settings_VS
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

        # Initialize overlay with configured screen and bind to screen changes
        screen_name = settings_state.screen
        if not screen_name:
            raise RuntimeError("No screen configured in settings")
        screen = self.get_screen_by_name(screen_name)
        if screen is None:
            raise RuntimeError(f"Configured screen '{screen_name}' not found")
        self._qt_app.bind_screen_overlay(screen)

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


from assistant.inference.context import ContextItem


class ContextController:
    """Handles context item CRUD for client operations and publishes updates on client-updates."""

    def __init__(self, facade_ref: "Facade") -> None:
        self._facade = facade_ref

    def next_position(self) -> int:
        self._ensure_session_started()
        return self._facade.context_manager.next_position()

    def create(self, item: ContextItem) -> Change:
        self._ensure_session_started()
        change = self._facade.context_manager.insert(item)
        self._broadcast(change)
        return change

    def update(self, item: ContextItem) -> Change:
        self._ensure_session_started()
        change = self._facade.context_manager.update(item)
        self._broadcast(change)
        return change

    def delete(self, item: ContextItem) -> Change:
        self._ensure_session_started()
        change = self._facade.context_manager.remove(item)
        self._broadcast(change)
        return change

    def _broadcast(self, change: Change) -> None:
        self._facade.app_state.context_VS.apply_change(change)
        self._facade.client_updates.push(change)

    def _ensure_session_started(self) -> None:
        session_state = self._facade.app_state.settings_VS.session_state
        if session_state != "started":
            raise RuntimeError("Cannot modify context without an active session.")


def apply_inference_event(evt):
    log.info(f"Applying inference event: {evt}")
    ctx_mgr = facade.context_manager
    app_ctx_view = facade.app_state.context_VS
    if isinstance(evt, Change):
        if evt.is_create() and isinstance(evt.new_state, ContextItem):
            repo_change = ctx_mgr.insert(evt.new_state)
        elif evt.is_delete() and isinstance(evt.old_state, ContextItem):
            repo_change = ctx_mgr.remove(evt.old_state)
        elif evt.new_state and isinstance(evt.new_state, ContextItem):
            repo_change = ctx_mgr.update(evt.new_state)
        else:
            return
        app_ctx_view.apply_change(repo_change)
        facade.project_manager.hybrid_segment_service.handle_context_change()
    elif isinstance(evt, dict) and evt.get("type") == "AppendItemContentText":
        payload = evt["payload"]
        item_id = payload["item_id"]
        text = payload["text"]
        for existing in ctx_mgr.items_sorted():  # TODO: optimize lookup
            if str(existing.id) == str(item_id):
                if not isinstance(existing.content, dict):
                    existing.content = {}
                prior = existing.content.get("text", "")
                existing.content["text"] = f"{prior}{text}"
                repo_change = ctx_mgr.update(existing)
                app_ctx_view.apply_change(repo_change)
                facade.project_manager.hybrid_segment_service.handle_context_change()
                break
    else:
        return


facade = Facade()
