from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from assistant.facade import vii
from assistant.inference.context import ContextItem, ContextManager
from assistant.services.context_sync_client import ContextSyncClient
from assistant.services.hybrid_segment_service import HybridSegmentService
from assistant.util import get_screen_by_name

from .session_recorder import SessionRecorder, SessionRecorderConfig

TASK_FILENAME = "task.md"
SYSTEM_PROMPT_FILENAME = "system_prompt.md"
SESSIONS_DIRNAME = "sessions"
DEFAULT_WEBSOCKET_URL = "ws://desk:8008/ws/context"
# DEFAULT_WEBSOCKET_URL = "ws://127.0.0.1:8000/ws/context"

# Resolve inference websocket URL at import time (env override if provided, fallback to default)
INFERENCE_WS_URL = (
    os.environ.get("VII_ASSISTANT_WS_URL", DEFAULT_WEBSOCKET_URL).strip()
    or DEFAULT_WEBSOCKET_URL
)

# HTTP base URL derived from the websocket URL
_ws_base = INFERENCE_WS_URL.split("/ws/")[0]
INFERENCE_HTTP_BASE = _ws_base.replace("ws://", "http://", 1).replace(
    "wss://", "https://", 1
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

        # Context store (on_changes wired in facade)
        self.context_manager = context_manager

        # Session & inference runtime
        self._session_manager: Optional[SessionManager] = None
        self._context_sync_client: Optional[ContextSyncClient] = None
        self._running = False
        # Segment/vision service (always present)

        self.hybrid_segment_service = HybridSegmentService()

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
        settings_state = vii.app_state.settings_VS
        if settings_state.session_state == "started":
            return self._session_manager or self.create_session()

        # Ensure session manager
        if self._session_manager is None:
            self._session_manager = manager or self.create_session()
            meta = self._session_manager.metadata
            try:
                vii.qt_app.terminal_state.output_text = (
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
            try:
                target_screen = vii.current_watched_screen()
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

        # Set session state to "started" before adding context items
        settings_state.session_state = "started"

        # Ensure inference client
        if self._context_sync_client is None:
            ws_url = INFERENCE_WS_URL
            settings_state.post_info_message(
                f"Connecting to inference websocket at {ws_url}..."
            )
            store = self.context_manager._repo
            self._context_sync_client = ContextSyncClient(url=ws_url, store=store)
            self._context_sync_client.set_settings_state(settings_state)
            self._context_sync_client.start()

            # Block until the sync handshake completes. The receiver gets
            # full_state from the server (which clears the local store),
            # so ALL items must be added after this point.
            if not self._context_sync_client.wait_ready(timeout=5.0):
                settings_state.post_info_message(
                    "Warning: sync handshake did not complete in time"
                )

            # Add system prompt as first context item
            system_prompt_text = self.current_system_prompt()
            if system_prompt_text and system_prompt_text.strip():
                system_item = ContextItem.create_text(
                    position=0,
                    text=system_prompt_text.strip(),
                    origin="system",
                )
                vii.context_controller.create(system_item)

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
        settings_state = vii.app_state.settings_VS
        if settings_state.session_state != "started":
            return
        if self._session_manager and self._session_manager.is_recording:
            self._session_manager.stop_recording()
        self._running = False
        settings_state.session_state = new_state
        print("Project manager paused session")

    def new_session(self) -> None:
        settings_state = vii.app_state.settings_VS
        if self._session_manager and self._session_manager.is_recording:
            self._session_manager.stop_recording()
        # Stop old inference client so a fresh one is created on next start
        if self._context_sync_client:
            self._context_sync_client.stop()
            self._context_sync_client = None
        self._session_manager = self.create_session()
        meta = self._session_manager.metadata

        # --- Reset UI + context state ---
        app_state = vii.app_state
        terminal_state = vii.qt_app.terminal_state

        # Clear context repository and propagate deletions via store.on_changes
        vii.context_controller.clear()

        # Clear info/status messages
        settings_state.clear_info_messages()

        # Clear overlay shapes
        vii.app_state.overlay_VS.clear()

        # Reset terminal output
        terminal_state.output_text = (
            f"New session directory ready: {meta.session_id}\n{meta.path}"
        )
        # Reset request/progress
        settings_state.request_in_progress = False

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
        return vii.app_state.settings_VS.system_prompt_markdown or ""

    def is_running(self) -> bool:
        return self._running
