from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sivkit.libs.procedure import procedure
from sivkit.storage.websockets_client_sync import WebSocketsClientSync

from assistant.facade import vii
from assistant.inference.context import ContextManager, TextMessage
from assistant.procedures import handle_hybrid_context_delta
from assistant.services.hybrid_segment_service import HybridSegmentService
from assistant.util import get_screen_by_name

from .session_recorder import SessionRecorder, SessionRecorderConfig

TASK_FILENAME = "task.md"
SYSTEM_PROMPT_FILENAME = "system_prompt.md"
SESSIONS_DIRNAME = "sessions"


@dataclass(slots=True)
class SessionMetadata:
    """Lightweight descriptor for a recorded session."""

    session_id: str
    path: Path
    created_at: datetime


class SessionManager:
    """Coordinates recording and persistence for a single session."""

    ACTIVITY_LOG_FILENAME = "activity_log.jsonl"

    def __init__(
        self,
        metadata: SessionMetadata,
        recorder_config: Optional[SessionRecorderConfig] = None,
    ) -> None:
        self.metadata = metadata
        self._recording = False
        self._recorder_config = recorder_config
        self._recorder: Optional[SessionRecorder] = None

    @classmethod
    def create(
        cls,
        sessions_root: Path | str,
        recorder_config: Optional[SessionRecorderConfig] = None,
    ) -> "SessionManager":
        metadata = cls.create_metadata(sessions_root)
        return cls(metadata=metadata, recorder_config=recorder_config)

    @classmethod
    def create_metadata(cls, sessions_root: Path | str) -> SessionMetadata:
        root = Path(sessions_root)
        root.mkdir(parents=True, exist_ok=True)

        created_at = datetime.now(timezone.utc)
        session_id = created_at.strftime("%Y-%m-%d_%H-%M-%S")
        session_path = root / session_id
        # Append counter suffix on collision (rapid clicks)
        counter = 1
        while session_path.exists():
            session_id = f"{created_at.strftime('%Y-%m-%d_%H-%M-%S')}_{counter}"
            session_path = root / session_id
            counter += 1
        session_path.mkdir(parents=True, exist_ok=False)

        return SessionMetadata(
            session_id=session_id,
            path=session_path,
            created_at=created_at,
        )

    # Public API ---------------------------------------------------------
    def start_recording(
        self, recorder_config: Optional[SessionRecorderConfig] = None
    ) -> None:
        if self._recording:
            return
        if recorder_config is not None:
            self._recorder_config = recorder_config
        config = self._recorder_config
        self._recorder = SessionRecorder(
            event_callback=self.record_event,
            config=config,
        )
        self._recorder.start()
        self._recording = True

    def stop_recording(self) -> None:
        if not self._recording or not self._recorder:
            return
        self._recorder.stop()
        self._recording = False
        self._recorder = None

    def record_event(self, entry: dict[str, Any]) -> None:
        if not isinstance(entry, dict):
            raise TypeError("Session activity entries must be dictionaries")

        entry_with_metadata = {
            **entry,
            "session_id": self.metadata.session_id,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        log_path = self.metadata.path / self.ACTIVITY_LOG_FILENAME
        with log_path.open("a", encoding="utf-8") as handle:
            json.dump(entry_with_metadata, handle)
            handle.write("\n")
        # print(f"[SessionManager] activity_log entry: {json.dumps(entry_with_metadata)}")

    @property
    def session_id(self) -> str:
        return self.metadata.session_id

    @property
    def session_path(self) -> Path:
        return self.metadata.path

    @property
    def is_recording(self) -> bool:
        return self._recording


class ViiProjectManager:
    """Project + automation manager (merged former AutomationService responsibilities).

    Responsibilities:
    - Persist task & system prompt
    - Manage sessions (create/start/stop/new)
    - Own context manager reference (facade delegates through here)
    - Host websocket inference client (WebSocketSyncService receiver)
    - Provide current_system_prompt accessor
    """

    hybrid_segment_service: HybridSegmentService  # always present after __init__

    def __init__(
        self, project_root: Path | str, *, context_manager: ContextManager
    ) -> None:
        self.project_root = Path(project_root)
        self.project_root.mkdir(parents=True, exist_ok=True)

        self._task_path = self.project_root / TASK_FILENAME
        self._system_prompt_path = self.project_root / SYSTEM_PROMPT_FILENAME
        self._sessions_root = self.project_root / SESSIONS_DIRNAME

        # Context store
        self.context_manager = context_manager

        # Session & inference runtime
        self._session_manager: Optional[SessionManager] = None
        self._sync_client: Optional[WebSocketsClientSync] = None
        self._running = False
        # Segment/vision service (always present)

        self.hybrid_segment_service = HybridSegmentService()

        # Wire hybrid context processing to store changes
        def _on_hybrid(delta, origin=None):
            _ = handle_hybrid_context_delta(self.hybrid_segment_service, delta, origin)

        context_manager._store.add_on_changes_callback(_on_hybrid)

        # Write-authority guard: raises ValueError if this process (the client)
        # writes a server-owned field.
        from assistant.inference.authority import authority_guard

        context_manager._store.add_on_changes_callback(
            lambda d, o: authority_guard("vii-assistant", d, o)
        )

    # Task/system prompt -------------------------------------------------
    def set_task(self, task: str) -> None:
        self._write_markdown(self._task_path, task)

    def get_task(self) -> Optional[str]:
        return self._read_markdown(self._task_path)

    def set_system_prompt(self, prompt: str) -> None:
        self._write_markdown(self._system_prompt_path, prompt)

    def get_system_prompt(self) -> Optional[str]:
        return self._read_markdown(self._system_prompt_path)

    # Sessions -----------------------------------------------------------
    def create_session(self) -> SessionManager:
        """Create a session and return its manager without starting recording."""
        metadata = SessionManager.create_metadata(self._sessions_root)
        return SessionManager(metadata=metadata)

    @procedure
    async def start_session(
        self,
        *,
        recorder_config: Optional[SessionRecorderConfig] = None,
        manager: Optional[SessionManager] = None,
        screen_name: str | None = None,
    ) -> SessionManager:
        settings_state = vii.app.view_state.settings_VS
        if settings_state.session_state == "started":
            return self._session_manager or self.create_session()

        # Ensure session manager
        if self._session_manager is None:
            self._session_manager = manager or self.create_session()
            meta = self._session_manager.metadata
            try:
                vii.app.terminal_state.output_text = (
                    f"Session directory ready: {meta.session_id}\n{meta.path}"
                )
            except Exception:
                pass

        # Build recorder config dynamically (screen geometry etc)
        config: SessionRecorderConfig = recorder_config or {}
        if screen_name:
            config["screen"] = screen_name
        target_screen = None
        if screen_name:
            target_screen = get_screen_by_name(screen_name)
        if target_screen is None:
            capture = vii.app.view_state.capture_screen_info
            if capture:
                target_screen = get_screen_by_name(capture.name)
        if target_screen is not None and hasattr(target_screen, "geometry"):
            geom = target_screen.geometry()  # type: ignore[call-arg]
            config["window_geometry"] = (
                geom.x(),
                geom.y(),
                geom.width(),
                geom.height(),
            )

        if self._session_manager is not None:
            pass  # Recording disabled — VCS persistence replaces it

        # Ensure inference client
        if self._sync_client is None:
            ws_url = vii.inference_client.ws_url
            settings_state.post_info_message(
                f"Connecting to inference websocket at {ws_url}..."
            )
            store = self.context_manager._store
            self._sync_client = WebSocketsClientSync(ws_url, store, role="receiver")

            # Connect + handshake. Returns once store is hydrated.
            try:
                await self._sync_client.connect(timeout=5.0)
            except Exception as exc:
                settings_state.session_state = "error"
                settings_state.post_info_message(
                    f"Error connecting to inference server: {exc}"
                )
                self._sync_client = None
                return self._session_manager

            settings_state.post_info_message(
                f"Connected to inference websocket at {ws_url}"
            )

            # React to disconnects
            self._sync_client.done.add_done_callback(
                lambda _: self._on_sync_disconnected()
            )

            # Connection established — now mark session as started
            settings_state.session_state = "started"

            # Insert system prompts for all focus modes that have prompt files
            from assistant.inference.focus_modes import FOCUS_MODES

            for mode_name, mode_config in FOCUS_MODES.items():
                if not mode_config.has_prompt_file:
                    continue
                try:
                    prompt_text = mode_config.load_system_prompt(vii.active_agent)
                except (FileNotFoundError, TypeError):
                    continue
                if not prompt_text.strip():
                    continue
                system_item = TextMessage()
                system_item.position = self.context_manager.next_position()
                system_item.text = prompt_text.strip()
                system_item.origin = "system"
                system_item.metadata = {"focus_mode": mode_name}
                vii.project_manager.context_manager.insert(system_item)

        if not self._running:
            self._running = True
            try:
                prompt_preview = self.current_system_prompt().strip().splitlines()[0:1]
            except Exception:
                prompt_preview = []
            preview = (
                f" | prompt: {prompt_preview[0][:60]}…"
                if prompt_preview and prompt_preview[0]
                else ""
            )
            print(f"Project manager started automation (session + inference){preview}")
        return self._session_manager

    def pause_session(self, *, new_state: str = "paused") -> None:
        settings_state = vii.app.view_state.settings_VS
        if settings_state.session_state != "started":
            return
        if self._session_manager and self._session_manager.is_recording:
            self._session_manager.stop_recording()
        self._running = False
        settings_state.session_state = new_state
        print("Project manager paused session")

    def _on_sync_disconnected(self) -> None:
        settings_state = vii.app.view_state.settings_VS
        if settings_state.session_state == "started":
            settings_state.session_state = "disconnected"
            settings_state.post_info_message("Inference websocket connection closed")

    def new_session(self) -> None:
        settings_state = vii.app.view_state.settings_VS
        if self._session_manager and self._session_manager.is_recording:
            self._session_manager.stop_recording()
        # Stop old inference client so a fresh one is created on next start
        if self._sync_client:
            self._sync_client.stop()
            self._sync_client = None
        self._session_manager = self.create_session()
        meta = self._session_manager.metadata

        # --- Reset UI + context state ---
        app_state = vii.app.view_state
        terminal_state = vii.app.terminal_state

        # Clear context repository and propagate deletions via store.on_changes
        vii.project_manager.context_manager.clear()

        # Reset agentic turn counter
        self.hybrid_segment_service.reset_turns()

        # Clear info/status messages
        settings_state.clear_info_messages()

        # Clear overlay shapes
        vii.app.view_state.overlay_VS.clear()

        # Reset terminal output
        terminal_state.output_text = (
            f"New session directory ready: {meta.session_id}\n{meta.path}"
        )
        # Reset request/progress
        settings_state.assistant_working = False

        settings_state.session_state = "new-session"
        self._running = False
        print("Project manager prepared new session")

        # Auto-restart session if inference server is available
        if vii.app.view_state.inference_status_VS.model_state == "loaded":
            capture = vii.app.view_state.capture_screen_info
            screen_name = capture.name if capture else vii.get_config().capture_screen
            self.start_session(screen_name=screen_name)

    @property
    def sessions_root(self) -> Path:
        """Expose the root directory where session data is stored."""
        return self._sessions_root

    # Internals ----------------------------------------------------------
    def _write_markdown(self, path: Path, content: str) -> None:
        normalized = content if content.endswith("\n") else f"{content}\n"
        path.write_text(normalized, encoding="utf-8")

    def _read_markdown(self, path: Path) -> Optional[str]:
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    # --- Automation helpers -------------------------------------------
    def current_system_prompt(self) -> str:
        return vii.app.view_state.settings_VS.system_prompt_markdown or ""

    def is_running(self) -> bool:
        return self._running
