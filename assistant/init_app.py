"""One-shot app initialization — services, view models, and Qt app for the assistant."""

from assistant.app_state import AppState
from assistant.experiments_manager import ExperimentsManager
from assistant.qml_app import AssistantQmlApp
from assistant.services.audio_recording import AudioRecordingService
from assistant.services.recording_overlay_view_model import RecordingOverlayViewModel
from assistant.services.settings_modal_view_model import SettingsModalViewModel
from assistant.services.snippet_view_model import SnippetViewModel
from assistant.services.transcription_chunking import (
    CHUNK_DURATION_S,
    OVERLAP_DURATION_S,
)
from assistant.services.transcription_orchestrator import TranscriptionOrchestrator
from assistant.snippet_procedures import handle_snippet_result


def init_app(facade) -> AssistantQmlApp:
    """Instantiate AppState, all services/view models, and the Qt app.

    Must be called after set_project_manager().
    """
    app_state = AppState()
    facade._app_state = app_state
    facade._experiments_manager = ExperimentsManager(facade)
    facade._recording_overlay_view_model = RecordingOverlayViewModel(app_state)
    facade._audio_recording_service = AudioRecordingService(
        overlay_view_model=facade._recording_overlay_view_model,
        chunk_duration_s=CHUNK_DURATION_S,
        overlap_duration_s=OVERLAP_DURATION_S,
    )
    facade._transcription_orchestrator = TranscriptionOrchestrator()
    facade._settings_modal_view_model = SettingsModalViewModel(app_state)
    facade._snippet_view_model = SnippetViewModel(app_state)
    facade._snippet_view_model.region_captured.connect(handle_snippet_result)

    qt_app = AssistantQmlApp(app_state)
    facade.set_qt_app(qt_app)
    return qt_app
