from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import pytesseract
from PIL import Image
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QGuiApplication, QPixmap, QScreen

from assistant.config import CLIENT_CONFIG, DEFAULT_CLIENT_TYPE, Config
from assistant.services.automation import AutomationService
from assistant.services.base_client import BaseClient
from assistant.services.ollama_client import OllamaClient
from assistant.services.project_manager import SessionManager, ViiProjectManager
from assistant.services.session_recorder import SessionRecorderConfig
from assistant.services.vllm_client import VLLMClient
from assistant.util import decode_client_type

if TYPE_CHECKING:
    from assistant.qt_app import AssistantQtApp


class Facade:
    _qt_app = None
    _desktop_server = None
    _config = None
    _automation_service = None
    _model_client: Optional[BaseClient] = None
    _watched_screen = None
    # Track (backend, model) to avoid unnecessary client re-inits
    _model_signature: Optional[Tuple[str, str]] = None
    _project_manager: Optional[ViiProjectManager] = None
    _session_manager: Optional[SessionManager] = None

    def __init__(self):
        # Setup config
        self._config = Config()
        self._config.set_config_changed_callback(self.apply_config)

        self._automation_service = AutomationService()
        # Initialize client from current config (source of truth)
        self.set_model_client(self._config.get("client_type", DEFAULT_CLIENT_TYPE))
        # Initialize project storage in the configuration directory by default
        self._project_manager = ViiProjectManager(self._config.config_dir)
        self._session_manager = None

    @property
    def qt_app(self) -> "AssistantQtApp":
        if self._qt_app is None:
            raise RuntimeError("Qt app instance not set")
        return self._qt_app

    @property
    def config(self) -> "Config":
        if self._config is None:
            raise RuntimeError("Config instance not initialized")
        return self._config

    @property
    def automation(self) -> AutomationService:
        if self._automation_service is None:
            raise RuntimeError("Automation service not initialized")
        return self._automation_service

    @property
    def inference_client(self) -> BaseClient:
        if self._model_client is None:
            raise RuntimeError("Client not initialized")
        return self._model_client

    @property
    def project_manager(self) -> ViiProjectManager:
        if self._project_manager is None:
            raise RuntimeError("Project manager not initialized")
        return self._project_manager

    def set_model_client(self, client_type: Optional[str] = None) -> None:
        """Initialize the appropriate client based on the client_type from config/state."""
        if client_type is None:
            client_type = DEFAULT_CLIENT_TYPE

        # Parse the client_type string
        backend, model = decode_client_type(client_type)

        # Validate backend
        if backend not in CLIENT_CONFIG:
            raise ValueError(f"Unknown backend type: {backend}")

        # Validate model
        if model not in CLIENT_CONFIG[backend]:
            raise ValueError(f"Unknown model '{model}' for backend '{backend}'")

        # No-op if signature hasn't changed
        if self._model_signature == (backend, model) and self._model_client is not None:
            return

        if backend == "ollama":
            self._model_client = OllamaClient("http://desk:11434")
        elif backend == "vllm":
            self._model_client = VLLMClient("http://desk:8000", model)
        else:
            raise ValueError(f"Unknown backend type: {backend}")

        self._model_signature = (backend, model)

    # Facade no longer stores client_type; use config/terminal state instead.

    @property
    def watched_screen(self) -> Optional[QScreen]:
        """Get the watched screen."""
        return self._watched_screen

    def setQtApp(self, app: "AssistantQtApp"):
        self._qt_app = app

        terminal_state = self._qt_app.terminal_state

        # Connect state changes from the terminal to the config manager
        terminal_state.system_prompt_changed.connect(
            lambda prompt: self.config.set("system_prompt", prompt)
        )
        terminal_state.client_type_changed.connect(
            lambda value: self.config.set("client_type", value)
        )
        terminal_state.screen_changed.connect(
            lambda value: self.config.set("screen", value)
        )
        model_settings = self._qt_app.terminal_window.model_settings
        model_settings.single_step_clicked.connect(
            lambda: self.single_step(source="screen")
        )
        model_settings.clipboard_step_clicked.connect(
            lambda: self.single_step(source="clipboard")
        )
        model_settings.ocr_clipboard_clicked.connect(self.ocr_clipboard_action)
        model_settings.start_session_clicked.connect(self.start_session)
        model_settings.stop_session_clicked.connect(self.stop_session)
        model_settings.new_session_clicked.connect(self.new_session)
        model_settings.open_sessions_folder_clicked.connect(self.open_sessions_folder)

        # Update terminal window with current config
        terminal_state.update_from_config(self.config._config)

    def start_desktop_server(self, port: int):
        """Start the desktop server."""
        from assistant.server.desktop_server import DesktopServer

        server = DesktopServer(port)
        server.start()
        self._desktop_server = server

    def apply_config(self, config: Dict[str, Any]):
        """Apply configuration changes."""
        # Session state is no longer persisted in config; control via explicit methods

        # Handle system_prompt setting
        system_prompt = config.get("system_prompt", "")
        self.automation.set_system_prompt(system_prompt)

        # Handle client_type setting
        client_type = config.get("client_type", DEFAULT_CLIENT_TYPE)
        try:
            backend, model = decode_client_type(client_type)
        except Exception:
            backend = model = None  # Invalid client_type; ignore
        else:
            if self._model_signature != (backend, model):
                self.set_model_client(client_type)

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

        self.qt_app.terminal_state.update_from_config(config)

    def start_session(self) -> None:
        """Start or resume the automation-backed session."""
        state = self.qt_app.terminal_state.session_state
        if state == "started":
            return

        if self._session_manager is None:
            self._session_manager = self.project_manager.create_session()
            metadata = self._session_manager.metadata
            message = f"Session directory ready: {metadata.session_id}\n{metadata.path}"
            print(message)
            self.qt_app.terminal_state.output_text = message

        session_config: SessionRecorderConfig = {}
        screen_name = self.qt_app.terminal_state.screen
        if screen_name:
            session_config["screen"] = screen_name

        target_screen: Optional[QScreen] = None
        if screen_name:
            target_screen = self.get_screen_by_name(screen_name)
        if target_screen is None:
            target_screen = self.watched_screen
        if target_screen is None:
            try:
                target_screen = self.get_default_screen()
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
        client_type = self.qt_app.terminal_state.client_type
        if client_type:
            session_config["client_type"] = client_type

        config_arg: Optional[SessionRecorderConfig] = session_config or None
        self._session_manager.start_recording(config_arg)

        if not self.automation.is_running():
            self.automation.start()

        self.qt_app.terminal_state.session_state = "started"

    def pause_session(self) -> None:
        """Pause the active session without resetting state."""
        if self.qt_app.terminal_state.session_state != "started":
            return
        if self.automation.is_running():
            self.automation.stop()
        if self._session_manager and self._session_manager.is_recording:
            self._session_manager.stop_recording()
        self.qt_app.terminal_state.session_state = "paused"

    def stop_session(self) -> None:
        """Stop the current session entirely."""
        if self.qt_app.terminal_state.session_state != "started":
            return
        if self.automation.is_running():
            self.automation.stop()
        if self._session_manager and self._session_manager.is_recording:
            self._session_manager.stop_recording()
        self.qt_app.terminal_state.session_state = "paused"

    def new_session(self) -> None:
        """Create a fresh session workspace on disk."""
        if self._session_manager and self._session_manager.is_recording:
            self._session_manager.stop_recording()
        if self.automation.is_running():
            self.automation.stop()

        self._session_manager = self.project_manager.create_session()
        metadata = self._session_manager.metadata

        message = f"New session directory ready: {metadata.session_id}\n{metadata.path}"
        print(message)
        self.qt_app.terminal_state.session_state = "new-session"
        self.qt_app.terminal_state.output_text = message

    def open_sessions_folder(self) -> None:
        """Open the sessions directory in the system file browser."""
        sessions_dir = self.project_manager.sessions_root
        sessions_dir.mkdir(parents=True, exist_ok=True)

        url = QUrl.fromLocalFile(str(sessions_dir))
        opened = QDesktopServices.openUrl(url)
        if not opened:
            print(f"Failed to open sessions directory: {sessions_dir}")

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
        terminal_state = self.qt_app.terminal_state

        # Get image first to check if it's available
        if source == "screen":
            pixmap = self.take_screenshot()
            if not pixmap:
                error_msg = "Failed to take screenshot"
                print(error_msg)
                terminal_state.output_text = error_msg
                return
        elif source == "clipboard":
            pixmap = self.get_clipboard_image()
            if not pixmap:
                error_msg = "No image found in clipboard"
                print(error_msg)
                terminal_state.output_text = error_msg
                return
        else:
            error_msg = f"Unknown source: {source}"
            print(error_msg)
            terminal_state.output_text = error_msg
            return

        # Update the image in the settings widget (on main thread)
        settings_widget.set_image(pixmap)

        # Set request in progress BEFORE starting worker thread
        terminal_state.request_in_progress = True

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
            self.qt_app.terminal_state.output_text = msg
            return

        # Indicate progress
        settings_widget = self.qt_app.terminal_window.model_settings
        self.qt_app.terminal_state.request_in_progress = True

        self.ocr_worker = OCRWorker(pixmap)
        self.ocr_worker.finished.connect(self._on_ocr_finished)
        self.ocr_worker.error.connect(self._on_worker_error)
        self.ocr_worker.start()

    def _on_ocr_finished(self, text: str):
        self.qt_app.terminal_state.request_in_progress = False
        self.qt_app.terminal_state.output_text = text

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
        self.qt_app.terminal_state.request_in_progress = False

        response, system_prompt, image_width, image_height = result

        # Check if response starts with "Error:" which indicates an error
        if response.startswith("Error:"):
            print(f"API error: {response}")
            self.qt_app.terminal_state.output_text = response
        else:
            # Format and print the response
            formatted_response = self.inference_client.format_response(
                system_prompt, response
            )
            print(f"Model response: {formatted_response}")
            self.qt_app.terminal_state.output_text = formatted_response

            # Use the client's extract_shapes method
            shapes = self.inference_client.extract_shapes(
                response, image_width, image_height
            )

            # Update the overlay with the shapes
            if shapes:
                self.qt_app.overlay.setShapes(shapes)
            else:
                self.qt_app.overlay.setShapes([])

    def _on_worker_error(self, error_msg):
        """Handle error from VLM processing."""
        self.qt_app.terminal_state.request_in_progress = False
        print(error_msg)
        self.qt_app.terminal_state.output_text = error_msg
        self.qt_app.overlay.hide()


class VLMWorker(QThread):
    """Worker thread for VLM processing to avoid blocking the UI."""

    finished = Signal(tuple)  # (response, system_prompt, screen_width, screen_height)
    error = Signal(str)  # error message

    def __init__(self, facade: Facade, source, pixmap: QPixmap):
        super().__init__()
        self.facade = facade
        # self.source = source
        self.pixmap = pixmap

    def run(self):
        """Run VLM processing in background thread."""
        try:
            # Get the system prompt
            system_prompt = self.facade.automation.get_system_prompt()

            # Convert pixmap to bytes
            buffer = QBuffer()
            buffer.open(QIODevice.OpenModeFlag.WriteOnly)
            try:
                self.pixmap.save(buffer, "PNG")

                # Call the VLM with the image
                # Derive model from config's client_type
                client_type = self.facade.config.get("client_type", DEFAULT_CLIENT_TYPE)
                _, model_name = decode_client_type(client_type)
                print(f"Calling VLM model '{model_name}' with prompt: {system_prompt}")
                response = self.facade.inference_client.call_vlm(
                    model_name, system_prompt, bytes(buffer.data().data())
                )

                # # Get image dimensions for coordinate scaling
                # # For screen source, use screen dimensions
                # # For clipboard source, use actual image dimensions
                # if self.source == "screen" and self.facade.watched_screen:
                #     image_width = self.facade.watched_screen.size().width()
                #     image_height = self.facade.watched_screen.size().height()
                # else:
                #     # Use pixmap dimensions for clipboard images
                #     image_width = self.pixmap.width()
                #     image_height = self.pixmap.height()

                # Emit success signal with results
                self.finished.emit(
                    (response, system_prompt, self.pixmap.height(), self.pixmap.width())
                )
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
