"""Transcription model manager for chunk-level inference.

Chunking and stitching are intentionally handled on the client side.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from assistant.transcription import ParakeetTranscriber, TranscriptionResult

# Default cache location for models
MODELS_CACHE_DIR = Path.home() / ".cache" / "vii-assistant" / "models"

# Model registry - maps model type string to config
TRANSCRIPTION_MODELS = {
    "parakeet-tdt-0.6b-v3-int8": {
        "class": ParakeetTranscriber,
        "default_dir": MODELS_CACHE_DIR / "parakeet-tdt-0.6b-v3-int8",
    },
}


def _detect_device() -> str:
    """Detect CUDA availability without importing torch at module level."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class TranscriptionService:
    """Manages transcription models and serves chunk-level inference."""

    def __init__(self, device: str | None = None):
        self._device = device or _detect_device()
        self._model: ParakeetTranscriber | None = None
        self._model_type: str | None = None

    @property
    def loaded_model(self) -> str | None:
        return self._model_type

    def load_model(self, model_type: str, model_dir: str | None = None) -> None:
        """Load a transcription model (or reuse if already loaded)."""
        if self._model_type == model_type and self._model is not None:
            return

        if model_type not in TRANSCRIPTION_MODELS:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {list(TRANSCRIPTION_MODELS.keys())}"
            )

        spec = TRANSCRIPTION_MODELS[model_type]
        dir_path = Path(model_dir or spec["default_dir"])

        self._model = spec["class"](model_dir=dir_path, device=self._device)
        self._model_type = model_type

    def transcribe_chunk(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        model_type: str = "parakeet-tdt-0.6b-v3-int8",
        model_dir: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe one audio chunk.

        The server performs only model inference. Client code is responsible
        for chunking long audio and stitching multi-chunk outputs.
        """
        self.load_model(model_type, model_dir)
        assert self._model is not None
        return self._model.transcribe(audio, sample_rate)

    # Backward-compat alias. New code should call transcribe_chunk().
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        model_type: str = "parakeet-tdt-0.6b-v3-int8",
        model_dir: str | None = None,
    ) -> TranscriptionResult:
        return self.transcribe_chunk(audio, sample_rate, model_type, model_dir)

    def transcribe_file(
        self,
        path: str | Path,
        model_type: str = "parakeet-tdt-0.6b-v3-int8",
        model_dir: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe an audio file."""
        import soundfile as sf

        audio, sr = sf.read(str(path), dtype="float32")
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return self.transcribe_chunk(audio, sr, model_type, model_dir)
