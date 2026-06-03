"""Persistence adapter for AppConfig: debounced flush to JSON file.

Attach to an InMemoryStore via add_on_changes_callback.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from sivkit import get_logger
from sivkit.storage.delta import Delta

log = get_logger(__name__)

CONFIG_DIR = Path.home() / ".config" / "vii-assistant"
CONFIG_FILE = CONFIG_DIR / "config.json"
DEBOUNCE_SECONDS = 1.0


class ConfigFileAdapter:
    """Debounced JSON persistence for the config InMemoryStore."""

    def __init__(self, store) -> None:
        self._store = store
        self._debounce_timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def on_store_changed(self, delta: Delta, origin: str | None = None) -> None:
        """Callback for InMemoryStore.add_on_changes_callback."""
        with self._lock:
            if self._debounce_timer is not None:
                self._debounce_timer.cancel()
            self._debounce_timer = threading.Timer(DEBOUNCE_SECONDS, self._flush)
            self._debounce_timer.daemon = True
            self._debounce_timer.start()

    def _flush(self) -> None:
        """Write current config to disk."""
        with self._lock:
            self._debounce_timer = None

        entity = self._store.find_one(id="app-config")
        if entity is None:
            return

        data = {
            "capture_screen": entity.capture_screen,
            "selected_model": entity.selected_model,
            "max_new_tokens": entity.max_new_tokens,
            "transcription": entity.transcription,
        }

        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            CONFIG_FILE.write_text(json.dumps(data, indent=2))
        except IOError as e:
            log.error("Failed to write config: %s", e)
