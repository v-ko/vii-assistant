import os
import signal
import sys
from importlib.metadata import PackageNotFoundError, version

# Set logging level before importing anything else (especially fusion)
os.environ.setdefault("LOGLEVEL", "INFO")

import click

from assistant.server.client import port_is_taken, send_command

DEFAULT_DESKTOP_SERVER_PORT = 51177


try:
    fusion_pkg_version = version("python-fusion")
except PackageNotFoundError:
    fusion_pkg_version = None
if not fusion_pkg_version or not fusion_pkg_version.startswith("0.1"):
    raise RuntimeError("Required fusion version >=0.10")


@click.command()
@click.option("--command", help="Command to send to the running instance")
def main(command):
    """Screenshot Assistant with HTTP API support."""
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # If a command is specified, try to send it to a running instance
    # Check if an instance is already running
    if port_is_taken(DEFAULT_DESKTOP_SERVER_PORT):
        print(f"An instance is already running on port {DEFAULT_DESKTOP_SERVER_PORT}.")

        if command:
            success = send_command(DEFAULT_DESKTOP_SERVER_PORT, command)
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

    from assistant.facade import vii
    from assistant.inference.context import ContextManager
    from assistant.init_app import init_app
    from assistant.model_configs import DEFAULT_MODEL_KEY, MODEL_SPECS
    from assistant.server.desktop_server import DesktopServer
    from assistant.services.project_manager import ViiProjectManager

    # Instantiate services first then inject into facade to avoid circular imports
    ctx_manager = ContextManager()
    vii.set_project_manager(
        ViiProjectManager(vii.config.config_dir, context_manager=ctx_manager)
    )

    qt_app = init_app(vii)

    # Configure image preprocessor model (loaded lazily on first use)
    vii.set_image_preprocessor_config(MODEL_SPECS[DEFAULT_MODEL_KEY]["id"])

    # Config already loaded; facade ensures screen is set during set_qt_app
    print(f"Config: {vii.config}")

    # Auto-load the configured model on the inference server
    from assistant.terminal_actions import set_model

    set_model(vii.app_state.settings_VS.selected_model)

    # Start desktop server as independent service
    desktop_server = DesktopServer(DEFAULT_DESKTOP_SERVER_PORT)
    desktop_server.start()

    # If this is a first start and --command was provided, execute it.
    if command:
        from assistant.recording_procedures import toggle_recording
        from assistant.terminal_actions import toggle_terminal

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
