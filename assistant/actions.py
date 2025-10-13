from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices, QScreen

from assistant.services.session_recorder import SessionRecorderConfig


def start_session(facade, *, screen_name: str | None = None) -> None:
    settings_state = facade.app_state.settings
    if settings_state.session_state == "started":
        return
    if facade._session_manager is None:  # noqa: SLF001
        facade._session_manager = (
            facade.project_manager.create_session()
        )  # noqa: SLF001
        metadata = facade._session_manager.metadata  # noqa: SLF001
        message = f"Session directory ready: {metadata.session_id}\n{metadata.path}"
        print(message)
        facade.qt_app.terminal_state.output_text = message

    session_config: SessionRecorderConfig = {}
    if screen_name:
        session_config["screen"] = screen_name

    target_screen: Optional[QScreen] = None
    if screen_name:
        target_screen = facade.get_screen_by_name(screen_name)
    if target_screen is None:
        target_screen = facade.watched_screen
    if target_screen is None:
        try:
            target_screen = facade.get_default_screen()
        except Exception:
            target_screen = None

    if target_screen is not None:
        geometry = target_screen.geometry()
        session_config["window_geometry"] = (
            geometry.x(),
            geometry.y(),
            geometry.width(),
            geometry.height(),
        )

    config_arg: Optional[SessionRecorderConfig] = session_config or None
    facade._session_manager.start_recording(config_arg)  # noqa: SLF001
    if not facade.automation.is_running():
        facade.automation.start()
    settings_state.session_state = "started"


def pause_or_stop_session(facade, *, new_state: str) -> None:
    settings_state = facade.app_state.settings
    if settings_state.session_state != "started":
        return
    if facade.automation.is_running():
        facade.automation.stop()
    if facade._session_manager and facade._session_manager.is_recording:  # noqa: SLF001
        facade._session_manager.stop_recording()  # noqa: SLF001
    settings_state.session_state = new_state


def new_session(facade) -> None:
    if facade._session_manager and facade._session_manager.is_recording:  # noqa: SLF001
        facade._session_manager.stop_recording()  # noqa: SLF001
    if facade.automation.is_running():
        facade.automation.stop()
    facade._session_manager = facade.project_manager.create_session()  # noqa: SLF001
    metadata = facade._session_manager.metadata  # noqa: SLF001
    message = f"New session directory ready: {metadata.session_id}\n{metadata.path}"
    print(message)
    facade.app_state.settings.session_state = "new-session"
    facade.qt_app.terminal_state.output_text = message


def open_sessions_folder(facade) -> None:
    sessions_dir: Path = facade.project_manager.sessions_root
    sessions_dir.mkdir(parents=True, exist_ok=True)
    url = QUrl.fromLocalFile(str(sessions_dir))
    opened = QDesktopServices.openUrl(url)
    if not opened:
        print(f"Failed to open sessions directory: {sessions_dir}")
