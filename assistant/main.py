import signal
import sys
import click
from assistant.actions import execute_action
from assistant.server.util import send_command
from assistant.server.util import port_is_taken
from PySide6.QtGui import QGuiApplication

DEFAULT_DESKTOP_SERVER_PORT = 51177


@click.command()
@click.option('--command', help='Command to send to the running instance')
def main(command):
    """Screenshot Assistant with HTTP API support."""
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # If a command is specified, try to send it to a running instance

    # Check if an instance is already running
    if port_is_taken(DEFAULT_DESKTOP_SERVER_PORT):
        print(f"An instance is already running on port "
              f"{DEFAULT_DESKTOP_SERVER_PORT}.")

        if command:
            success = send_command(DEFAULT_DESKTOP_SERVER_PORT, command)
            sys.exit(0 if success else 1)

        else:
            from PySide6.QtWidgets import QMessageBox
            from PySide6.QtWidgets import QApplication
            tmp_app = QApplication(sys.argv)
            messageBox = QMessageBox()
            messageBox.critical(
                messageBox, "Error",
                f"An instance is already running on port "
                f"{DEFAULT_DESKTOP_SERVER_PORT}.")
            tmp_app.exec()

        sys.exit(1)

    from assistant.qt_app import AssistantQtApp
    from assistant.facade import facade

    # Start a new instance
    qt_app = AssistantQtApp()
    facade.setQtApp(qt_app)

    # Initialize config and set screen
    config = facade.config
    print(f"Config: {config}")

    # If screen is not set or not available, set to default
    screen_name = config.get("screen", "")
    screen = None
    if screen_name:
        screen = facade.get_screen_by_name(screen_name)

    if not screen:
        # Set default screen
        default_screen = facade.get_default_screen()
        if default_screen:
            config.set("screen", default_screen.name())

    # Update terminal window from config
    facade.apply_config(facade.config.data())

    # Start desktop server
    facade.start_desktop_server(DEFAULT_DESKTOP_SERVER_PORT)

    # If this is a first start and --command was provided,
    # show the terminal directly
    if command:
        execute_action(command)

    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
