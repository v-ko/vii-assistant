from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import pytesseract
from PIL import Image
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QThread, Signal
from PySide6.QtGui import QGuiApplication, QPixmap, QScreen

from assistant.config import CLIENT_CONFIG
from assistant.services.automation import AutomationService
from assistant.services.base_client import BaseClient
from assistant.services.ollama_client import OllamaClient
from assistant.services.vllm_client import VLLMClient
from assistant.util import decode_client_type

if TYPE_CHECKING:
    from assistant.config import Config
    from assistant.qt_app import AssistantQtApp


class Facade:
    _qt_app = None
    _desktop_server = None
    _config = None
    _automation_service = None
    _client: Optional[BaseClient] = None
    _client_type: str = "ollama:moondream"
    _watched_screen = None

    def __init__(self):
        self._automation_service = AutomationService()
        self._init_client()

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

    def _parse_client_type(self) -> Tuple[str, str]:
        """Parse the client_type string into backend and model.

        Validates that the backend and model are valid according to CLIENT_CONFIG.
        If not, defaults to a known good configuration.

        Returns:
            A tuple of (backend, model)
        """
        # Default values
        default_backend = "ollama"
        default_model = "moondream"

        # Parse the client_type string
        backend, model = decode_client_type(self._client_type)

        # Validate backend
        if backend not in CLIENT_CONFIG:
            print(f"Unknown backend: {backend}, using default")
            return default_backend, default_model

        # Validate model
        if model not in CLIENT_CONFIG[backend]:
            print(f"Model {model} not available for backend {backend}, using default")
            # Use the first model in the list for this backend
            model = CLIENT_CONFIG[backend][0]

        return backend, model

    def _init_client(self) -> None:
        """Initialize the appropriate client based on the client_type."""
        backend, model = self._parse_client_type()

        if backend == "ollama":
            self._client = OllamaClient("http://desk:11434")
        elif backend == "vllm":
            self._client = VLLMClient("http://desk:8000", model)
        else:
            raise ValueError(f"Unknown backend type: {backend}")

    @property
    def client(self) -> BaseClient:
        if self._client is None:
            raise RuntimeError("Client not initialized")
        return self._client

    @property
    def client_type(self) -> str:
        return self._client_type

    @property
    def backend(self) -> str:
        """Get the backend part of the client_type."""
        return self._parse_client_type()[0]

    @property
    def model(self) -> str:
        """Get the model part of the client_type."""
        return self._parse_client_type()[1]

    def set_client_type(self, client_type: str) -> None:
        """Set the client type and initialize the appropriate client."""
        if client_type != self._client_type:
            self._client_type = client_type
            self._init_client()

    @property
    def watched_screen(self) -> Optional[QScreen]:
        """Get the watched screen."""
        return self._watched_screen

    def setQtApp(self, app: "AssistantQtApp"):
        self._qt_app = app

        # Connect signals from terminal window
        self._qt_app.terminal_window.system_prompt_changed.connect(
            lambda prompt: self.config.set("system_prompt", prompt)
        )
        self._qt_app.terminal_window.config_changed.connect(
            lambda key, value: self.config.set(key, value)
        )
        self._qt_app.terminal_window.model_settings.single_step_clicked.connect(
            lambda: self.single_step(source="screen")
        )
        self._qt_app.terminal_window.model_settings.clipboard_step_clicked.connect(
            lambda: self.single_step(source="clipboard")
        )
        self._qt_app.terminal_window.model_settings.ocr_clipboard_clicked.connect(
            self.ocr_clipboard_action
        )

        # Update terminal window with current config
        self._qt_app.terminal_window.update_from_config(self.config._config)

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

        # Handle client_type setting
        client_type = config.get("client_type", "ollama:moondream")
        if client_type != self._client_type:
            self.set_client_type(client_type)

        # Handle screen setting
        screen_name = config.get("screen", "")
        current_screen = self.watched_screen.name() if self.watched_screen else ""

        if screen_name and screen_name != current_screen:
            screen = self.get_screen_by_name(screen_name)
            if screen:
                self._watched_screen = screen
                print(f"Watched screen set: {screen.name()}")
                # Update overlay to match the watched screen
                if self._qt_app and self._qt_app.overlay:
                    self._qt_app.overlay.setScreen(screen)  # Set to new screen
                    self._qt_app.overlay._setup_full_screen()  # Resize to fit screen
            else:
                # If screen not found, use default
                self._watched_screen = self.get_default_screen()
                print(f"Watched screen set to default: {self._watched_screen.name()}")
                # Update overlay to match the default screen
                if self._qt_app and self._qt_app.overlay:
                    self._qt_app.overlay.setScreen(
                        self._watched_screen
                    )  # Set to default screen
                    self._qt_app.overlay._setup_full_screen()  # Resize to fit screen

        self.qt_app.terminal_window.update_from_config(config)

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

    def get_clipboard_image(self) -> Optional[QPixmap]:
        """Get image from system clipboard.

        Returns:
            A QPixmap containing the clipboard image, or None if no image is found
        """
        try:
            clipboard = QGuiApplication.clipboard()
            mime_data = clipboard.mimeData()

            if mime_data.hasImage():
                # Get the image data and convert to QPixmap
                image_data = mime_data.imageData()
                if image_data:
                    pixmap = QPixmap(image_data)
                    return pixmap

            return None
        except Exception as e:
            print(f"Error getting clipboard image: {e}")
            return None

    def single_step(self, source: str = "screen") -> None:
        """Perform a single step of the automation process.

        This takes a screenshot or clipboard image, processes it with the VLM client,
        and updates the UI accordingly.

        Args:
            source: Either 'screen' for screenshot or 'clipboard' for clipboard image
        """
        # Get the settings widget
        settings_widget = self.qt_app.terminal_window.model_settings

        # Get image first to check if it's available
        if source == "screen":
            pixmap = self.take_screenshot()
            if not pixmap:
                error_msg = "Failed to take screenshot"
                print(error_msg)
                self.qt_app.terminal_window.set_output_text(error_msg)
                return
        elif source == "clipboard":
            pixmap = self.get_clipboard_image()
            if not pixmap:
                error_msg = "No image found in clipboard"
                print(error_msg)
                self.qt_app.terminal_window.set_output_text(error_msg)
                return
        else:
            error_msg = f"Unknown source: {source}"
            print(error_msg)
            self.qt_app.terminal_window.set_output_text(error_msg)
            return

        # Update the image in the settings widget (on main thread)
        settings_widget.set_image(pixmap)

        # Set request in progress BEFORE starting worker thread
        settings_widget.request_in_progress = True

        # Create and start worker thread with the pixmap
        self.worker = VLMWorker(self, source, pixmap)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

    def ocr_clipboard_action(self):
        """Perform OCR on the clipboard image and display text output.

        Uses local Tesseract via pytesseract. Non-blocking (worker thread).
        """
        pixmap = self.get_clipboard_image()
        if not pixmap:
            msg = "No image found in clipboard for OCR"
            print(msg)
            self.qt_app.terminal_window.set_output_text(msg)
            return

        # Indicate progress
        settings_widget = self.qt_app.terminal_window.model_settings
        settings_widget.request_in_progress = True

        self.ocr_worker = OCRWorker(pixmap)
        self.ocr_worker.finished.connect(self._on_ocr_finished)
        self.ocr_worker.error.connect(self._on_worker_error)
        self.ocr_worker.start()

    def _on_ocr_finished(self, text: str):
        settings_widget = self.qt_app.terminal_window.model_settings
        settings_widget.request_in_progress = False
        self.qt_app.terminal_window.set_output_text(text)

    # --- Getter API for external callers ---
    def get_clipboard_ocr_text(self, lang: str = "eng") -> str:
        """Get OCR text from the current clipboard image synchronously.

        Args:
            lang: Tesseract language code (default "eng").

        Returns:
            Recognized text, or an error message starting with "Error:".
        """
        pixmap = self.get_clipboard_image()
        if not pixmap:
            return "Error: No image in clipboard"
        try:
            # Convert pixmap to bytes
            ba = QByteArray()
            buf = QBuffer(ba)
            buf.open(QIODevice.OpenModeFlag.WriteOnly)
            try:
                pixmap.save(buf, "PNG")
            finally:
                buf.close()
            img_bytes = ba.data()  # sip.voidptr -> Python buffer interface
            pil_img = Image.open(io.BytesIO(bytes(img_bytes)))
            text = pytesseract.image_to_string(pil_img, lang=lang)
            return text.strip() or "<No text recognized>"
        except Exception as e:  # pragma: no cover
            return f"Error: OCR failed: {e}"

    def _on_worker_finished(self, result):
        """Handle successful completion of VLM processing."""
        settings_widget = self.qt_app.terminal_window.model_settings
        settings_widget.request_in_progress = False

        response, system_prompt, image_width, image_height = result

        # Check if response starts with "Error:" which indicates an error
        if response.startswith("Error:"):
            print(f"API error: {response}")
            self.qt_app.terminal_window.set_output_text(response)
        else:
            # Format and print the response
            formatted_response = self.client.format_response(system_prompt, response)
            print(f"Model response: {formatted_response}")
            self.qt_app.terminal_window.set_output_text(formatted_response)

            # Use the client's extract_shapes method
            shapes = self.client.extract_shapes(response, image_width, image_height)

            # Update the overlay with the shapes
            if shapes:
                self.qt_app.overlay.setShapes(shapes)
            else:
                self.qt_app.overlay.setShapes([])

    def _on_worker_error(self, error_msg):
        """Handle error from VLM processing."""
        settings_widget = self.qt_app.terminal_window.model_settings
        settings_widget.request_in_progress = False
        print(error_msg)
        self.qt_app.terminal_window.set_output_text(error_msg)
        self.qt_app.overlay.hide()


class VLMWorker(QThread):
    """Worker thread for VLM processing to avoid blocking the UI."""

    finished = Signal(tuple)  # (response, system_prompt, screen_width, screen_height)
    error = Signal(str)  # error message

    def __init__(self, facade, source, pixmap):
        super().__init__()
        self.facade = facade
        self.source = source
        self.pixmap = pixmap

    def run(self):
        """Run VLM processing in background thread."""
        try:
            # Get the system prompt
            system_prompt = (
                self.facade.automation.get_system_prompt() or "What's in this image?"
            )

            # Convert pixmap to bytes
            buffer = QBuffer()
            buffer.open(QIODevice.OpenModeFlag.WriteOnly)
            try:
                self.pixmap.save(buffer, "PNG")

                # Call the VLM with the image
                response = self.facade.client.call_vlm(
                    self.facade.model, system_prompt, bytes(buffer.data().data())
                )

                # Get image dimensions for coordinate scaling
                # For screen source, use screen dimensions
                # For clipboard source, use actual image dimensions
                if self.source == "screen" and self.facade.watched_screen:
                    image_width = self.facade.watched_screen.size().width()
                    image_height = self.facade.watched_screen.size().height()
                else:
                    # Use pixmap dimensions for clipboard images
                    image_width = self.pixmap.width()
                    image_height = self.pixmap.height()

                # Emit success signal with results
                self.finished.emit((response, system_prompt, image_width, image_height))
            finally:
                # Always close the buffer
                buffer.close()
        except Exception as e:
            # Handle any unexpected exceptions
            error_msg = f"Error during processing: {e}"
            self.error.emit(error_msg)


class OCRWorker(QThread):
    """Worker thread for performing Tesseract OCR on a QPixmap."""

    finished = Signal(str)  # OCR text
    error = Signal(str)

    def __init__(self, pixmap: QPixmap, lang: str = "eng"):
        super().__init__()
        self.pixmap = pixmap
        self.lang = lang

    def run(self):  # type: ignore[override]
        try:
            # Convert QPixmap to PNG bytes via QBuffer/QByteArray
            ba = QByteArray()
            buffer = QBuffer(ba)
            buffer.open(QIODevice.OpenModeFlag.WriteOnly)
            try:
                # Use QPixmap.save directly (simpler & reliable)
                self.pixmap.save(buffer, "PNG")
            finally:
                buffer.close()
            raw = ba.data()
            pil_image = Image.open(io.BytesIO(bytes(raw)))
            text = pytesseract.image_to_string(pil_image, lang=self.lang)
            self.finished.emit((text.strip()) or "<No text recognized>")
        except Exception as e:  # pragma: no cover
            self.error.emit(f"Error during OCR: {e}")


facade = Facade()
