import json
import pathlib
from typing import Any, Callable, Dict, List, Optional

from assistant.services import qwen25_preprocess

# Client configuration mapping backends to available models
CLIENT_CONFIG = {
    "ollama": ["moondream", "gemma3", "qwen2.5vl"],
    "vllm": ["osunlp/UGround-V1-2B"],
}

# Hardcoded model-to-extractor key routing per backend (no regex)
# Keys in the inner dict must exactly match model names in CLIENT_CONFIG.
MODEL_EXTRACTOR_MAP = {
    "ollama": {
        "qwen2.5vl": qwen25_preprocess.extract_qwen25_shapes_ollama_policy,
    },
    "vllm": {
        # Add vLLM model-specific extractors here if needed
    },
}


class Config:
    """Configuration manager for the assistant application."""

    def __init__(self):
        self.config_dir = pathlib.Path.home() / ".config" / "vii-assistant"
        self.config_file = self.config_dir / "config.json"
        self.config_changed_callback: Optional[Callable[[Dict[str, Any]], None]] = None

        # Default configuration
        self._config = {
            "auto_query": False,
            "screen": "",  # Will be set to actual screen in init
            "system_prompt": "",
            "client_type": "ollama:moondream",  # Default client type with model
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

        # Notify about the change
        if self.config_changed_callback:
            self.config_changed_callback(self._config)

    def set_config_changed_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set the callback to be called when the configuration changes."""
        self.config_changed_callback = callback
