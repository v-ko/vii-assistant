from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from assistant.facade import facade
from assistant.inference.context import ContextManager
from assistant.services.inference_client import InferenceClient

from .session_recorder import SessionRecorder, SessionRecorderConfig

TASK_FILENAME = "task.md"
SYSTEM_PROMPT_FILENAME = "system_prompt.md"
SESSIONS_DIRNAME = "sessions"
WEBSOCKET_URL_ENV_VAR = "VII_ASSISTANT_WS_URL"
# DEFAULT_WEBSOCKET_URL = "ws://desk:8008/ws/context"
DEFAULT_WEBSOCKET_URL = "ws://127.0.0.1:8000/ws/context"

# Resolve inference websocket URL at import time (env override if provided, fallback to default)
INFERENCE_WS_URL = (
    os.environ.get(WEBSOCKET_URL_ENV_VAR, DEFAULT_WEBSOCKET_URL).strip()
    or DEFAULT_WEBSOCKET_URL
)


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
        print(f"[SessionManager] activity_log entry: {json.dumps(entry_with_metadata)}")

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
    - Host websocket inference client + channels (client_updates / inference_updates)
    - Provide current_system_prompt accessor
    - Wire inference updates to reducer (apply_context_event)
    """

    def __init__(
        self, project_root: Path | str, *, context_manager: ContextManager
    ) -> None:
        self.project_root = Path(project_root)
        self.project_root.mkdir(parents=True, exist_ok=True)

        self._task_path = self.project_root / TASK_FILENAME
        self._system_prompt_path = self.project_root / SYSTEM_PROMPT_FILENAME
        self._sessions_root = self.project_root / SESSIONS_DIRNAME

        # Context (channels now owned by facade)
        self.context_manager = context_manager

        # Session & inference runtime
        self._session_manager: Optional[SessionManager] = None
        self._inference_client: Optional[InferenceClient] = None
        self._running = False

        # Inference events wired in facade.set_project_manager

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

    def start_session(
        self,
        *,
        recorder_config: Optional[SessionRecorderConfig] = None,
        manager: Optional[SessionManager] = None,
        screen_name: str | None = None,
    ) -> SessionManager:
        settings_state = facade.app_state.settings_VS
        if settings_state.session_state == "started":
            return self._session_manager or self.create_session()

        # Ensure session manager
        if self._session_manager is None:
            self._session_manager = manager or self.create_session()
            meta = self._session_manager.metadata
            try:
                facade.qt_app.terminal_state.output_text = (
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
            target_screen = facade.get_screen_by_name(screen_name)
        if target_screen is None:
            try:
                target_screen = facade.current_watched_screen()
            except Exception:
                target_screen = None
        if target_screen is not None and hasattr(target_screen, "geometry"):
            geom = target_screen.geometry()  # type: ignore[call-arg]
            config["window_geometry"] = (
                geom.x(),
                geom.y(),
                geom.width(),
                geom.height(),
            )

        if self._session_manager is not None:
            self._session_manager.start_recording(config or None)

        # Ensure inference client
        if self._inference_client is None:
            ws_url = INFERENCE_WS_URL
            settings_state.post_info_message(
                f"Connecting to inference websocket at {ws_url}..."
            )
            self._inference_client = InferenceClient(url=ws_url)
            self._inference_client.start()
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
        settings_state.session_state = "started"
        return self._session_manager

    def pause_session(self, *, new_state: str = "paused") -> None:
        settings_state = facade.app_state.settings_VS
        if settings_state.session_state != "started":
            return
        if self._session_manager and self._session_manager.is_recording:
            self._session_manager.stop_recording()
        self._running = False
        settings_state.session_state = new_state
        print("Project manager paused session")

    def new_session(self) -> None:
        settings_state = facade.app_state.settings_VS
        if self._session_manager and self._session_manager.is_recording:
            self._session_manager.stop_recording()
        self._session_manager = self.create_session()
        meta = self._session_manager.metadata
        try:
            # --- Reset UI + context state ---
            app_state = facade.app_state
            terminal_state = facade.qt_app.terminal_state

            # Clear context repository and propagate deletions to view + client channel
            try:
                deletions = self.context_manager.clear()
                if deletions:
                    for ch in deletions:
                        # Apply to in-memory view + broadcast to client channel
                        app_state.context_VS.apply_change(ch)
                        facade.client_updates.push(ch)
            except Exception as e:  # pragma: no cover - defensive
                print(f"Failed to clear context on new session: {e}")

            # Clear info/status messages
            try:
                settings_state.clear_info_messages()
            except Exception:
                pass

            # Reset terminal output
            terminal_state.output_text = (
                f"New session directory ready: {meta.session_id}\n{meta.path}"
            )
            # Reset request/progress & disallow context edits until session started
            settings_state.request_in_progress = False
            settings_state.context_updates_allowed = False
        except Exception:
            pass
        settings_state.session_state = "new-session"
        print("Project manager prepared new session")

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
        return facade.app_state.settings_VS.system_prompt_markdown or ""

    def is_running(self) -> bool:
        return self._running
