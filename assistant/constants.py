"""Deployment constants resolved at import time."""

import os
from pathlib import Path

from PySide6.QtCore import QStandardPaths

# Load .env from project root (sibling of this package)
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
if _ENV_FILE.is_file():
    for _line in _ENV_FILE.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _key, _, _val = _line.partition("=")
        os.environ.setdefault(_key.strip(), _val.strip())

DEFAULT_SERVER_URL = "http://localhost:8008"

INFERENCE_HTTP_BASE = (
    os.environ.get("VII_ASSISTANT_SERVER", DEFAULT_SERVER_URL).strip()
    or DEFAULT_SERVER_URL
).rstrip("/")

# Derive websocket URL from HTTP base
INFERENCE_WS_URL = (
    INFERENCE_HTTP_BASE.replace("http://", "ws://", 1).replace("https://", "wss://", 1)
    + "/ws/context"
)

# App data paths
APP_DATA_DIR = Path(
    QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
)
RECORDINGS_DIR = APP_DATA_DIR / "transcription_recordings"
MAX_SAVED_RECORDINGS = 5

# Experiment configs
EXPERIMENTS_DIR = Path(__file__).parent / "experiments"
