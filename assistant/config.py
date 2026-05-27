import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional


class Config:
    """Configuration manager for the assistant application."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_dir = Path(config_path or Path.home() / ".config" / "vii-assistant")
        self.config_file = self.config_dir / "config.json"

        self.config_changed_callback: Optional[Callable[[Dict[str, Any]], None]] = None

        # Default configuration
        self._config = {
            "screen": "",  # Will be set to actual screen in init
            "system_prompt": "",
            "selected_model": "qwen3_vl_4b",
            "default_generation_params": {
                "max_new_tokens": 256,
                "temperature": 0.0,
            },
            "max_new_tokens": 256,
        }

        self._init_config()

    def __repr__(self):
        return f"Config({self._config})"

    def data(self) -> Dict[str, Any]:
        """Get the configuration data."""
        return self._config

    def _init_config(self):
        """Initialize the configuration file if it doesn't exist."""
        # Create config directory if it doesn't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Load config if it exists, otherwise create it
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    loaded_config = json.load(f)
                    self._config.update(loaded_config)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading config: {e}")
        else:
            self.save_config()

    def save_config(self):
        """Save the current configuration to the config file."""
        try:
            with open(self.config_file, "w") as f:
                json.dump(self._config, f, indent=2)
        except IOError as e:
            print(f"Error saving config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any):
        """Set a configuration value and save the configuration."""
        if key in self._config and self._config[key] == value:
            return  # No change
        self._config[key] = value
        self.save_config()
        if self.config_changed_callback:
            self.config_changed_callback(self._config)

    def set_config_changed_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set the callback to be called when the configuration changes."""
        self.config_changed_callback = callback

    # --- bulk update API (no per-key callbacks) -------------------
    def update_bulk(self, values: Dict[str, Any], *, notify: bool = True) -> None:
        """Apply multiple key/value updates with a single disk write.

        Existing keys are only written if changed. Keys not present in values are untouched.
        If nothing changes, no save or callback.
        """
        changed = False
        for k, v in values.items():
            if k not in self._config or self._config[k] != v:
                self._config[k] = v
                changed = True
        if not changed:
            return
        self.save_config()
        if notify and self.config_changed_callback:
            self.config_changed_callback(self._config)
