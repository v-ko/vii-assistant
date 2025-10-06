from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

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
    """Persist project-level assistant state to disk."""

    def __init__(
        self,
        project_root: Path | str,
    ) -> None:
        self.project_root = Path(project_root)
        self.project_root.mkdir(parents=True, exist_ok=True)

        self._task_path = self.project_root / TASK_FILENAME
        self._system_prompt_path = self.project_root / SYSTEM_PROMPT_FILENAME
        self._sessions_root = self.project_root / SESSIONS_DIRNAME

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
    ) -> SessionManager:
        session_manager = manager or self.create_session()
        session_manager.start_recording(recorder_config)
        return session_manager

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
