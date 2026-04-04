from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

from PySide6.QtCore import QObject

from assistant.config import Config
from assistant.view_states.settings import AssistantSettingsViewState


class ConfigPersistenceService(QObject):
    """Coalesce view-state changes into debounced config writes.

    Strategy:
    - Observe selected SettingsViewState signals.
    - Maintain a pending dict of new values.
    - Arm/reset a timer on each change; after debounce interval, flush via Config.update_bulk.
    """

    def __init__(self, parent: Optional[QObject] = None, debounce_seconds: float = 1):
        super().__init__(parent)
        self._config: Optional[Config] = None
        self._settings: Optional[AssistantSettingsViewState] = None
        self._pending: Dict[str, Any] = {}
        self._debounce_seconds = debounce_seconds
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None

    def bind(self, config: Config, settings: AssistantSettingsViewState) -> None:
        self._config = config
        self._settings = settings

        # connect signals
        settings.screen_changed.connect(lambda v: self._enqueue("screen", v))
        settings.selected_model_changed.connect(
            lambda v: self._enqueue("selected_model", v)
        )

    # --- internal -------------------------------------------------
    def _enqueue(self, key: str, value: Any) -> None:
        with self._lock:
            self._pending[key] = value
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce_seconds, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self) -> None:
        with self._lock:
            payload = self._pending
            self._pending = {}
            self._timer = None
        if not payload:
            return
        if not self._config:
            return
        self._config.update_bulk(payload)

    def flush_now(self) -> None:
        self._flush()
