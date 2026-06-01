"""One-shot app initialization — services, view models, and Qt app for the assistant."""

from assistant.app_state import AppViewState
from assistant.experiments_manager import ExperimentsManager
from assistant.qml_app import ViiQmlApp
from assistant.services.audio_recording import AudioRecordingService
from assistant.services.transcription_chunking import (
    CHUNK_DURATION_S,
    OVERLAP_DURATION_S,
)
from assistant.services.transcription_orchestrator import TranscriptionOrchestrator
from assistant.snippet_procedures import handle_snippet_result


def init_app(facade) -> ViiQmlApp:
    """Instantiate the Qt app, all services/view models, and wire projectors.

    Must be called after set_project_manager().
    """
    # Phase 1: Create app (minimal — no QML loaded yet)
    app_state = AppViewState()
    qt_app = ViiQmlApp(app_state)
    facade._app = qt_app  # facade available to services

    # Phase 2: Services (can access vii.app freely now)
    facade._experiments_manager = ExperimentsManager()

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
    facade._transcription_orchestrator = TranscriptionOrchestrator()

    # Cross-vertical wiring
    qt_app.snippet_view_model.region_captured.connect(handle_snippet_result)

    # Phase 3: Load UI (all services ready — QML slots won't crash)
    qt_app.init_ui()

    # Phase 4: Wire projectors
    facade.init_qt_app()
    return qt_app
