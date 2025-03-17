from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Dict, Any
from PySide6.QtGui import QScreen, QGuiApplication, QPixmap
from PySide6.QtCore import QBuffer, QIODevice

from assistant.services.automation import AutomationService
from assistant.services.ollama_client import OllamaClient

if TYPE_CHECKING:
    from assistant.qt_app import AssistantQtApp
    from assistant.config import Config

MODEL = 'moondream'  # moonderam


class Facade:
    _qt_app = None
    _desktop_server = None
    _config = None
    _automation_service = None
    _ollama_client = None
    _watched_screen = None

    def __init__(self):
        self._automation_service = AutomationService()
        self._ollama_client = OllamaClient("http://desk:11434")

    @property
    def qt_app(self) -> "AssistantQtApp":
        if self._qt_app is None:
            raise RuntimeError("Qt app instance not set")
        return self._qt_app

    @property
    def config(self) -> "Config":
        if self._config is None:
            from assistant.config import Config
            self._config = Config()
            self._config.set_config_changed_callback(self.apply_config)
        return self._config

    @property
    def automation(self) -> AutomationService:
        if self._automation_service is None:
            raise RuntimeError("Automation service not initialized")
        return self._automation_service

    @property
    def client(self) -> OllamaClient:
        if self._ollama_client is None:
            raise RuntimeError("Ollama client not initialized")
        return self._ollama_client

    @property
    def watched_screen(self) -> Optional[QScreen]:
        """Get the watched screen."""
        return self._watched_screen

    def setQtApp(self, app: "AssistantQtApp"):
        self._qt_app = app

    def start_desktop_server(self, port: int):
        """Start the desktop server."""
        from assistant.server.desktop_server import DesktopServer
        server = DesktopServer(port)
        server.start()
        self._desktop_server = server

    def apply_config(self, config: Dict[str, Any]):
        """Apply configuration changes."""
        # Check if service is running
        was_running = self.automation.is_running()

        # Handle auto_query setting
        auto_query = config.get("auto_query", False)
        if auto_query and not was_running:
            self.automation.start()
        elif not auto_query and was_running:
            self.automation.stop()

        # Handle system_prompt setting
        system_prompt = config.get("system_prompt", "")
        self.automation.set_system_prompt(system_prompt)

        # Handle screen setting
        screen_name = config.get("screen", "")
        current_screen = self.watched_screen.name(
        ) if self.watched_screen else ""

        if screen_name and screen_name != current_screen:
            screen = self.get_screen_by_name(screen_name)
            if screen:
                self._watched_screen = screen
                print(f"Watched screen set: {screen.name()}")
            else:
                # If screen not found, use default
                self._watched_screen = self.get_default_screen()
                print(
                    f"Watched screen set to default: {self._watched_screen.name()}"
                )

        self.qt_app.terminal_window.update_from_config()

    def get_screen_by_name(self, name: str) -> Optional[QScreen]:
        """Get a QScreen object by its name."""
        screens = QGuiApplication.screens()
        for screen in screens:
            if screen.name() == name:
                return screen
        return None

    def get_default_screen(self) -> QScreen:
        """Get the default screen (second to primary if available)."""
        screens = QGuiApplication.screens()
        if len(screens) > 1:
            primary = QGuiApplication.primaryScreen()
            screens.remove(primary)
        return screens[0]  # First non-primary screen

    def take_screenshot(self) -> Optional[QPixmap]:
        """Take a screenshot of the watched screen.

        Returns:
            A QPixmap containing the screenshot, or None if no screen is watched
            or if the screenshot fails.
        """
        if not self.watched_screen:
            print("No screen is being watched")
            return None

        try:
            # Take the screenshot
            pixmap = self.watched_screen.grabWindow(0)
            return pixmap
        except Exception as e:
            print(f"Error taking screenshot: {e}")
            return None

    def single_step(self) -> None:
        """Perform a single step of the automation process.

        This takes a screenshot, processes it with the ollama client,
        and updates the UI accordingly.
        """
        # Get the settings widget from the terminal window
        settings_widget = self.qt_app.terminal_window.model_settings

        # Set request in progress
        settings_widget.request_in_progress = True

        try:
            # Take screenshot
            pixmap = self.take_screenshot()
            if not pixmap:
                error_msg = "Failed to take screenshot"
                print(error_msg)
                self.qt_app.terminal_window.set_output_text(error_msg)
                settings_widget.request_in_progress = False
                return

            # Update the image in the settings widget
            settings_widget.set_image(pixmap)

            # Get the system prompt
            system_prompt = self.automation.get_system_prompt(
            ) or "What's in this image?"

            # Convert pixmap to bytes
            buffer = QBuffer()
            buffer.open(QIODevice.OpenModeFlag.WriteOnly)
            try:
                pixmap.save(buffer, "PNG")

                # Call the VLM with the screenshot
                response = self.client.call_vlm(MODEL, system_prompt,
                                                bytes(buffer.data().data()))

                # Check if response starts with "Error:" which indicates an error
                if response.startswith("Error:"):
                    print(f"API error: {response}")
                    # Still display the error to the user
                    self.qt_app.terminal_window.set_output_text(response)
                else:
                    # Format and print the response
                    formatted_response = self.client.format_response(
                        system_prompt, response)
                    print(f"Model response: {formatted_response}")

                    # Update the terminal window with the response
                    self.qt_app.terminal_window.set_output_text(
                        formatted_response)
            finally:
                # Always close the buffer
                buffer.close()
        except Exception as e:
            # Handle any unexpected exceptions
            error_msg = f"Error during processing: {str(e)}"
            print(error_msg)
            self.qt_app.terminal_window.set_output_text(error_msg)
        finally:
            # Always reset request in progress
            settings_widget.request_in_progress = False


facade = Facade()
