import os
import signal
import sys
from importlib.metadata import PackageNotFoundError, version

# Set logging level before importing anything else (especially fusion)
os.environ.setdefault("LOGLEVEL", "INFO")

import click
from PySide6.QtCore import QTimer

from assistant.registries.actions import execute_action
from assistant.server.client import port_is_taken, send_command

DEFAULT_DESKTOP_SERVER_PORT = 51177


try:
    fusion_pkg_version = version("python-fusion")
except PackageNotFoundError:
    fusion_pkg_version = None
if not fusion_pkg_version or not fusion_pkg_version.startswith("0.9"):
    raise RuntimeError("Required fusion version >=0.9")


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

    from assistant.facade import facade
    from assistant.inference.context import ContextManager
    from assistant.qt_app import AssistantQtApp
    from assistant.server.desktop_server import DesktopServer
    from assistant.services.ollama_client import OllamaClient
    from assistant.services.project_manager import ViiProjectManager

    # Instantiate services first then inject into facade to avoid circular imports
    ctx_manager = ContextManager()
    facade.set_project_manager(
        ViiProjectManager(facade.config.config_dir, context_manager=ctx_manager)
    )

    # Start a new instance (after services injected so set_qt_app can wire them)
    qt_app = AssistantQtApp()
    facade.set_qt_app(qt_app)

    # Configure image preprocessor model (loaded lazily on first use)
    from assistant.model_configs import MODEL_ID

    facade.set_image_preprocessor_config(MODEL_ID)

    # Config already loaded; facade ensures screen is set during set_qt_app
    print(f"Config: {facade.config}")

    # Start desktop server as independent service
    desktop_server = DesktopServer(DEFAULT_DESKTOP_SERVER_PORT)
    desktop_server.start()

    # If this is a first start and --command was provided,
    # show the terminal directly
    if command:
        execute_action(command)

    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
