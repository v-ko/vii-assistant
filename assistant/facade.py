from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Optional

import fusion
from fusion.storage.change import Change
from fusion.storage.delta import Delta
from PySide6.QtGui import QGuiApplication, QScreen

from assistant.app_state import AppState
from assistant.config import Config
from assistant.inference.context import ContextManager
from assistant.inference.context_store import OP_SEP, ContextStore
from assistant.services.config_persistence_service import ConfigPersistenceService
from assistant.util import get_screen_by_name

if TYPE_CHECKING:
    from assistant.qml_app import AssistantQmlApp
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
    _experiments_manager = None

    def __init__(self):
        self._config = Config()
        self._session_manager = None
        self.context_controller = ContextController(self)

    # --- explicit service setters (must be called early in main) ---------
    def set_project_manager(self, manager: ViiProjectManager):
        self._project_manager = manager
        # Wire store.on_changes to update view state + hybrid segment service
        store = manager.context_manager._repo
        store.on_changes = on_context_store_changes

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
    def qt_app(self) -> "AssistantQmlApp":
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

    @property
    def experiments_manager(self):
        if self._experiments_manager is None:
            from assistant.experiments_manager import ExperimentsManager

            self._experiments_manager = ExperimentsManager(self)
        return self._experiments_manager

    # Model/client selection removed; single client in use.

    def current_watched_screen(self) -> Optional[QScreen]:
        """Return the currently selected screen or raise if unavailable.

        No silent defaulting; caller must ensure a valid selection exists.
        """
        screen_name = self.app_state.settings_VS.screen
        if not screen_name:
            raise RuntimeError("No screen selected in settings")
        scr = get_screen_by_name(screen_name)
        if scr is None:
            raise RuntimeError(f"Configured screen '{screen_name}' not found")
        return scr

    def set_qt_app(self, app):
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
                screen_valid = get_screen_by_name(screen_name) is not None
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
        if get_screen_by_name(screen_name) is None:
            raise RuntimeError(f"Configured screen '{screen_name}' not found")
        self._qt_app.bind_screen_overlay(screen_name)

    def get_default_screen(self) -> QScreen:
        """Get the default screen (second to primary if available)."""
        screens = QGuiApplication.screens()
        if len(screens) > 1:
            primary = QGuiApplication.primaryScreen()
            screens.remove(primary)
        return screens[0]  # First non-primary screen


from assistant.inference.context import ContextItem


class ContextController:
    """Handles context item CRUD for client operations."""

    def __init__(self, facade_ref: "Facade") -> None:
        self._facade = facade_ref

    def next_position(self) -> int:
        self._ensure_session_started()
        return self._facade.context_manager.next_position()

    def create(self, item: ContextItem) -> Change:
        self._ensure_session_started()
        return self._facade.context_manager.insert(item)

    def update(self, item: ContextItem) -> Change:
        self._ensure_session_started()
        return self._facade.context_manager.update(item)

    def delete(self, item: ContextItem) -> Change:
        self._ensure_session_started()
        return self._facade.context_manager.remove(item)

    def clear(self) -> list[Change]:
        self._ensure_session_started()
        return self._facade.context_manager.clear()

    def _ensure_session_started(self) -> None:
        session_state = self._facade.app_state.settings_VS.session_state
        if session_state != "started":
            raise RuntimeError("Cannot modify context without an active session.")


def on_context_store_changes(delta: Delta, origin: str | None = None) -> None:
    """Callback wired to the client-side ContextStore.on_changes.

    May fire from any thread (sync client asyncio thread, daemon threads).
    Marshals the view state update to the Qt main thread via call_delayed.
    """
    keys = list(delta.asdict().keys())
    print(
        f"[TRACE] on_context_store_changes: origin={origin} keys={keys} thread={threading.current_thread().name}"
    )
    fusion.call_delayed(_apply_store_delta_to_view, 0, args=[delta])


def _apply_store_delta_to_view(delta: Delta) -> None:
    """Runs on the Qt main thread — safe to mutate QObject view states."""
    app_ctx_view = vii.app_state.context_VS
    ctx_mgr = vii.context_manager

    for key, change_data in delta.asdict().items():
        if OP_SEP in key:
            entity_id = key.split(OP_SEP, 1)[0]
            entity = ctx_mgr._repo.find_one(id=entity_id)
            if entity and isinstance(entity, ContextItem):
                app_ctx_view.apply_entity(entity)
            vii.project_manager.hybrid_segment_service.handle_context_change()
        else:
            eid, reverse, forward = change_data
            change = Change(eid, reverse, forward)
            if change.is_delete():
                app_ctx_view.remove_entity(eid)
            else:
                entity = ctx_mgr._repo.find_one(id=eid)
                if entity and isinstance(entity, ContextItem):
                    app_ctx_view.apply_entity(entity)
            vii.project_manager.hybrid_segment_service.handle_context_change()

            # Notify experiments manager (if active)
            if vii._experiments_manager is not None:
                vii._experiments_manager._on_inference_update(change)


vii = Facade()
