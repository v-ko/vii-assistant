from __future__ import annotations

import json
from typing import TYPE_CHECKING

from PySide6.QtGui import QGuiApplication, QScreen
from sivkit.storage.delta import Delta
from sivkit.storage.in_memory_store import InMemoryStore

from assistant.constants import AGENTS_DIR, EXPERIMENTS_DIR
from assistant.model.app_config import ViiConfig
from assistant.model.experiment_config import ExperimentConfig
from assistant.projections import project_config, project_context_delta_to_view
from assistant.services.config_file_adapter import CONFIG_FILE, ConfigFileAdapter
from assistant.services.inference_client import InferenceServerClient
from assistant.util import get_screen_by_name

if TYPE_CHECKING:
    from assistant.experiments_manager import ExperimentsManager
    from assistant.qml_app import ViiQmlApp
    from assistant.services.audio_recording import AudioRecordingService
    from assistant.services.project_manager import SessionManager, ViiProjectManager
    from assistant.services.transcription_orchestrator import TranscriptionOrchestrator

import logging

log = logging.getLogger(__name__)


class Facade:
    _app: ViiQmlApp | None = None
    _project_manager: ViiProjectManager | None = None
    _session_manager: SessionManager | None = None
    _experiments_manager: ExperimentsManager | None = None
    _audio_recording_service: AudioRecordingService | None = None
    _transcription_orchestrator: TranscriptionOrchestrator | None = None

    def __init__(self):
        self._config_store = InMemoryStore()
        self._config_file_adapter = ConfigFileAdapter(self._config_store)
        self._config_store.add_on_changes_callback(
            self._config_file_adapter.on_store_changed
        )
        self._load_config()
        self._session_manager = None
        self.inference_client = InferenceServerClient()
        self._active_agent: str | None = None

    def _load_config(self) -> None:
        """Load config from disk into the store."""
        config_data: dict = {}
        if CONFIG_FILE.exists():
            try:
                config_data = json.loads(CONFIG_FILE.read_text())
            except (json.JSONDecodeError, IOError) as e:
                log.error("Failed to read config: %s", e)

        entity = ViiConfig(
            capture_screen=config_data.get("capture_screen", ""),
            selected_model=config_data.get("selected_model", ""),
            max_new_tokens=int(config_data.get("max_new_tokens", 4096)),
            transcription=config_data.get("transcription", {"input_device": ""}),
        )
        self._config_store.insert_one(entity)

        # Load available experiment configs from filesystem
        for path in sorted(EXPERIMENTS_DIR.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, IOError) as e:
                log.error("Failed to read experiment config %s: %s", path, e)
                continue
            self._config_store.insert_one(
                ExperimentConfig(
                    id=ExperimentConfig.id_for_path(path.stem),
                    name=data.get("name", path.stem),
                    path=str(path),
                    data_loader=data.get("data_loader", ""),
                    prompt=data.get("prompt", ""),
                    dataset_path=data.get("dataset_path", ""),
                    prompt_template=data.get("prompt_template", ""),
                    generation_params=data.get("generation_params", {}),
                    resolution=data.get("resolution"),
                    start_index=data.get("start_index"),
                    end_index=data.get("end_index"),
                    stream=data.get("stream", True),
                    chat_template_params=data.get("chat_template_params", {}),
                    extraction=data.get("extraction", "response"),
                    focus_mode=data.get("focus_mode", "main"),
                    max_turns=data.get("max_turns") or {},
                )
            )

    # --- explicit service setters (must be called early in main) ---------
    def set_project_manager(self, manager: ViiProjectManager) -> None:
        self._project_manager = manager

    # --- Agent management -------------------------------------------------
    def get_available_agents(self) -> list[str]:
        """Return sorted list of agent folder names."""
        if not AGENTS_DIR.exists():
            return []
        return sorted(d.name for d in AGENTS_DIR.iterdir() if d.is_dir())

    @property
    def active_agent(self) -> str | None:
        """Return the currently active agent name, defaulting to the first available."""
        if self._active_agent is None:
            agents = self.get_available_agents()
            if agents:
                self._active_agent = agents[0]
        return self._active_agent

    def set_active_agent(self, name: str) -> None:
        """Set the active agent by folder name."""
        self._active_agent = name

    @property
    def app(self) -> ViiQmlApp:
        if self._app is None:
            raise RuntimeError("Qt app instance not set")
        return self._app

    def get_config(self) -> ViiConfig:
        cfg = self._config_store.find_one(id="app-config")
        assert cfg is not None
        return cfg

    def update_config(self, entity: ViiConfig) -> None:
        self._config_store.update_one(entity)

    def get_experiment_configs(self) -> list[ExperimentConfig]:
        """Return all available experiment configs from the store."""
        return list(self._config_store.find(type=ExperimentConfig))

    def get_experiment_config(self, config_id: str) -> ExperimentConfig | None:
        """Look up a single experiment config by ID."""
        return self._config_store.find_one(id=config_id)

    @property
    def project_manager(self) -> ViiProjectManager:
        if self._project_manager is None:
            raise RuntimeError(
                "Project manager not set; call setProjectManager in main"
            )
        return self._project_manager

    @property
    def experiments_manager(self) -> ExperimentsManager:
        if self._experiments_manager is None:
            raise RuntimeError(
                "Experiments manager not initialized; call init_app() first"
            )
        return self._experiments_manager

    @property
    def audio_recording_service(self) -> AudioRecordingService:
        if self._audio_recording_service is None:
            raise RuntimeError(
                "Audio recording service not initialized; call init_app() first"
            )
        return self._audio_recording_service

    @property
    def transcription_orchestrator(self) -> TranscriptionOrchestrator:
        if self._transcription_orchestrator is None:
            raise RuntimeError(
                "Transcription orchestrator not initialized; call init_app() first"
            )
        return self._transcription_orchestrator

    # Model/client selection removed; single client in use.

    def init_qt_app(self) -> None:
        if self._app is None:
            raise RuntimeError("Qt app must be created before init_qt_app")

        app_state = self._app.view_state
        settings_state = app_state.settings_VS

        # Register context store projector
        ctx_mgr = self.project_manager.context_manager
        context_vs = app_state.context_VS

        def _on_context_changed(delta: Delta, origin: str | None = None) -> None:
            project_context_delta_to_view(delta, origin, ctx_mgr, context_vs)

        ctx_mgr._store.add_on_changes_callback(_on_context_changed)

        # Register config projector — fires on every config store change
        settings_modal_vs = app_state.settings_modal_VS
        ts_vs = settings_modal_vs.transcription_section_vs

        def _on_config_changed(delta: Delta, origin: str | None = None) -> None:
            project_config(self.get_config(), settings_state, ts_vs)

        self._config_store.add_on_changes_callback(_on_config_changed)

        # Ensure a valid capture screen exists in config
        cfg = self.get_config()
        capture_screen = cfg.capture_screen
        if not capture_screen or get_screen_by_name(capture_screen) is None:
            default_screen = self.get_default_screen()
            capture_screen = default_screen.name()
            cfg.capture_screen = capture_screen
            self.update_config(cfg)  # triggers projector

        # Ensure a valid experiment config is selected
        if not settings_state.selected_experiment_config:
            configs = list(self._config_store.find(type=ExperimentConfig))
            if configs:
                settings_state.selected_experiment_config = configs[0].path

        # Run projector once for initial seeding
        project_config(self.get_config(), settings_state, ts_vs)

        settings_state._project_manager = self.project_manager
        settings_state.reload_project_documents()

        # Initialize overlay (view-layer setup)
        if not capture_screen:
            raise RuntimeError("No screen configured in settings")
        if get_screen_by_name(capture_screen) is None:
            raise RuntimeError(f"Configured screen '{capture_screen}' not found")
        self._app.bind_screen_overlay(capture_screen)

    def get_default_screen(self) -> QScreen:
        """Get the default screen (second to primary if available)."""
        screens = QGuiApplication.screens()
        if len(screens) > 1:
            primary = QGuiApplication.primaryScreen()
            screens.remove(primary)
        return screens[0]  # First non-primary screen


vii = Facade()


def raise_on_session_inactive() -> None:
    """Raise if the session is not started. Use at user-facing entry points."""
    state = vii.app.view_state.settings_VS.session_state
    if state != "started":
        raise RuntimeError(
            f"Cannot modify context: session is '{state}', not 'started'."
        )
