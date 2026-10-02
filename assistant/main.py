import time

_T_START = time.perf_counter()

import os
import signal
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

# Set logging level before importing anything else (especially sivkit)
os.environ.setdefault("LOGLEVEL", "INFO")

# Run via XWayland on Wayland sessions: overlays need stay-on-top/bypass-WM
if sys.platform.startswith("linux"):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import click

from assistant.server.command_client import port_is_taken, send_command

DEFAULT_DESKTOP_SERVER_PORT = 51177


try:
    sivkit_pkg_version = version("sivkit")
except PackageNotFoundError:
    sivkit_pkg_version = None
if not sivkit_pkg_version or not sivkit_pkg_version.startswith("0.1"):
    raise RuntimeError("Required sivkit version >=0.10")


@click.command()
@click.option("--command", help="Command to send to the running instance")
@click.option(
    "--measure-command-send-time",
    is_flag=True,
    hidden=True,
    help="Show notify-send with command send timing",
)
def main(command, measure_command_send_time):
    """Screenshot Assistant with HTTP API support."""
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # If a command is specified, try to send it to a running instance
    # Check if an instance is already running
    if port_is_taken(DEFAULT_DESKTOP_SERVER_PORT):
        print(f"An instance is already running on port {DEFAULT_DESKTOP_SERVER_PORT}.")

        if command:
            success = send_command(DEFAULT_DESKTOP_SERVER_PORT, command)
            if measure_command_send_time:
                elapsed_ms = (time.perf_counter() - _T_START) * 1000
                subprocess.Popen(
                    [
                        "notify-send",
                        "-t",
                        "3000",
                        "vii --command timing",
                        f"{command}: {elapsed_ms:.0f}ms",
                    ]
                )
            sys.exit(0 if success else 1)

        else:
            from PySide6.QtWidgets import QApplication, QMessageBox

            tmp_app = QApplication(
                sys.argv
            )  # Must be present for the message box to work
            messageBox = QMessageBox()
            messageBox.critical(
                messageBox,
                "Error",
                "An instance is already running on port "
                f"{DEFAULT_DESKTOP_SERVER_PORT}.",
            )

        sys.exit(1)

    from assistant.app.init_app import configure_logging, init_app
    from assistant.facade import vii
    from assistant.inference.context import ContextManager
    from assistant.server.desktop_server import DesktopServer
    from assistant.services.project_manager import ViiProjectManager

    configure_logging()

    # Instantiate services first then inject into facade to avoid circular imports
    ctx_manager = ContextManager()
    from assistant.services.config_file_adapter import CONFIG_DIR

    vii.set_project_manager(ViiProjectManager(CONFIG_DIR, context_manager=ctx_manager))

    qt_app = init_app(vii)

    print(f"Config: {vii.get_config()}")

    # Auto-load the configured model on the inference server
    from assistant.actions.terminal import set_model

    set_model(vii.get_config().selected_model)

    # Start desktop server as independent service
    desktop_server = DesktopServer(DEFAULT_DESKTOP_SERVER_PORT)
    desktop_server.start()

    if command:
        from assistant.actions.terminal import toggle_terminal
        from assistant.procedures.recording import toggle_recording

        fresh_start_commands = {
            "toggle_terminal": toggle_terminal,
            "toggle_recording": toggle_recording,
        }
        if command in fresh_start_commands:
            fresh_start_commands[command]()
        else:
            print(f"Command '{command}' ignored on fresh start")

    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
