"""Persistence adapter for AppConfig: debounced flush to JSON file.

Attach to an InMemoryStore via add_on_changes_callback.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from PySide6.QtCore import QTimer
from sivkit.storage.delta import Delta

log = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / ".config" / "vii-assistant"
CONFIG_FILE = CONFIG_DIR / "config.json"
DEBOUNCE_MS = 1000


class ConfigFileAdapter:
    """Debounced JSON persistence for the config InMemoryStore."""

    def __init__(self, store) -> None:
        self._store = store
        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(DEBOUNCE_MS)
        self._debounce_timer.timeout.connect(self._flush)

    def on_store_changed(self, delta: Delta, origin: str | None = None) -> None:
        """Callback for InMemoryStore.add_on_changes_callback."""
        self._debounce_timer.start()

    def _flush(self) -> None:
        """Write current config to disk."""
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
