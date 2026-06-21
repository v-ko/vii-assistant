"""One-shot app initialization — services, view models, and Qt app for the assistant."""

from assistant.app_state import AppViewState
from assistant.experiments_manager import ExperimentsManager
from assistant.headless_app import ViiHeadlessApp
from assistant.projections import project_inference_status
from assistant.qml_app import ViiQmlApp
from assistant.recording_actions import set_transcribing_active
from assistant.services.audio_recording import AudioRecordingService
from assistant.services.transcription_chunking import (
    CHUNK_DURATION_S,
    OVERLAP_DURATION_S,
)
from assistant.services.transcription_orchestrator import TranscriptionOrchestrator
from assistant.snippet_procedures import handle_snippet_result


def _wire_inference_status(
    facade, app_state, *, auto_start_session: bool, terminal_state=None
) -> None:
    """Connect the inference client's status to the view state.

    Projects status changes onto ``inference_status_VS`` and starts the
    service-owned poll loop. When *auto_start_session* is set, the loaded
    transition also starts a session (GUI policy). When *terminal_state* is
    provided, polling pauses while that terminal window is hidden.
    """
    client = facade.inference_client

    def _on_status_changed() -> None:
        prev_state = app_state.inference_status_VS.model_state
        project_inference_status(client, app_state)
        if not auto_start_session:
            return
        if client.model_state == "loaded" and prev_state != "loaded":
            settings = app_state.settings_VS
            if settings.session_state != "started":
                capture = app_state.capture_screen_info
                screen_name = capture.name if capture else ""
                facade.project_manager.start_session(screen_name=screen_name)

    client.set_on_status_changed(_on_status_changed)
    if terminal_state is not None:
        client.set_terminal_state(terminal_state)
    client.start_status_polling()


def init_app(facade) -> ViiQmlApp:
    """Instantiate the Qt app, all services/view models, and wire projectors.

    Must be called after set_project_manager().
    """
    # Phase 1: Create app (minimal — no QML loaded yet)
    app_state = AppViewState()
    qt_app = ViiQmlApp(app_state)
    facade._app = qt_app  # facade available to services

    # Phase 2: Services (can access vii.app freely now)
    facade._experiments_manager = ExperimentsManager(
        app_view_model=qt_app.app_view_model
    )

    cfg = facade.get_config()
    facade._audio_recording_service = AudioRecordingService(
        overlay_view_model=qt_app.recording_overlay_view_model,
        chunk_duration_s=CHUNK_DURATION_S,
        overlap_duration_s=OVERLAP_DURATION_S,
        selected_device_id=cfg.input_device,
    )
    # ARS reacts to input device changes via VS signal
    app_state.settings_modal_VS.transcription_section_vs.selected_input_device_changed.connect(
        lambda: setattr(
            facade._audio_recording_service,
            "selected_device_id",
            app_state.settings_modal_VS.transcription_section_vs.selected_input_device,
        )
    )
    facade._transcription_orchestrator = TranscriptionOrchestrator(
        facade.inference_client
    )
    facade._transcription_orchestrator.set_on_transcribing_changed(
        set_transcribing_active
    )

    # Cross-vertical wiring
    qt_app.snippet_view_model.region_captured.connect(handle_snippet_result)

    # Phase 3: Load UI (all services ready — QML slots won't crash)
    qt_app.init_ui()

    # Phase 4: Wire projectors
    facade.init_qt_app()

    # Phase 5: Inference status — service-owned polling + session policy
    _wire_inference_status(
        facade,
        app_state,
        auto_start_session=True,
        terminal_state=qt_app.terminal_state,
    )
    return qt_app


def init_headless(facade) -> ViiHeadlessApp:
    """Minimal init for headless experiment runs.

    Creates the GUI-free Qt app and the experiments manager only — no QML,
    services, or projectors. Must be called after set_project_manager().
    """
    app = ViiHeadlessApp(AppViewState())
    facade._app = app
    facade._experiments_manager = ExperimentsManager(app_view_model=app.app_view_model)
    # Service-owned status polling keeps inference_status_VS fresh; no session
    # policy in headless — the experiment runner waits on the model state itself.
    _wire_inference_status(facade, app.view_state, auto_start_session=False)
    return app
