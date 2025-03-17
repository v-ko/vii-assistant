import threading
import uvicorn

from fastapi import FastAPI
from assistant.server.util import port_is_taken
from assistant.server.routes import router


class DesktopServer:
    def __init__(self, port: int):
        self.thread = None
        self._port = port

    @property
    def port(self) -> int:
        return self._port

    def start(self):
        # Check if the port is already in use
        if port_is_taken(self.port):
            raise RuntimeError(f"Port {self.port} is already in use")

        # Create the FastAPI app
        app = FastAPI(title="Screenshot Assistant API")

        # Include the router
        app.include_router(router)

        # Configure the server
        config = uvicorn.Config(
            app=app,
            host="127.0.0.1",
            port=self.port,
            log_level="info"
        )
        self.server = uvicorn.Server(config=config)

        # Start the server in a thread
        self.thread = threading.Thread(target=self.server.run)
        self.thread.daemon = True
        self.thread.start()

        print(f"Server started at http://localhost:{self.port}")

    def stop(self):
        if self.thread and self.thread.is_alive():
            self.server.should_exit = True
            self.thread.join()
