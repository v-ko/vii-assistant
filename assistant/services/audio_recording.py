"""Audio recording service using QAudioSource.

Responsibilities:
- Start/stop recording on toggle
- Accumulate PCM in buffer (16kHz mono int16)
- Compute 10-band spectrogram at 10 Hz and push to pill controller
- Emit chunk-ready signals for transcription orchestrator
- Enforce 2h recording cap
"""

from __future__ import annotations

import struct
import time
from typing import TYPE_CHECKING

import numpy as np
from fusion import get_logger
from PySide6.QtCore import QByteArray, QIODevice, QObject, QTimer, Signal
from PySide6.QtMultimedia import QAudioFormat, QAudioSource, QMediaDevices

from assistant.recording_actions import set_recording_active
from assistant.services.transcription_chunking import (
    CHUNK_DURATION_S,
    OVERLAP_DURATION_S,
)

if TYPE_CHECKING:
    from assistant.services.recording_overlay_view_model import (
        RecordingOverlayViewModel,
    )

log = get_logger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_SIZE_BYTES = 2  # int16
MAX_RECORDING_SECONDS = 7200  # 2 hours hard cap
SPECTRUM_INTERVAL_MS = 100  # 10 Hz
SPECTRUM_WINDOW_SAMPLES = 800  # 50ms at 16kHz
NUM_BANDS = 10


class AudioRecordingService(QObject):
    """Manages audio capture from a configurable input device."""

    # Emitted when a chunk of audio is ready for transcription.
    # Args: (chunk_audio_int16: np.ndarray, chunk_start_time_s: float)
    chunk_ready = Signal(object, float)

    # Emitted when recording stops (final partial chunk if any).
    recording_stopped = Signal()

    def __init__(
        self,
        overlay_view_model: RecordingOverlayViewModel,
        *,
        chunk_duration_s: int,
        overlap_duration_s: int,
        selected_device_id: str = "",
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._overlay_vm = overlay_view_model
        self._chunk_duration_s = chunk_duration_s
        self._overlap_duration_s = overlap_duration_s
        self._selected_device_id = selected_device_id

        self._source: QAudioSource | None = None
        self._io_device: QIODevice | None = None
        self._buffer = QByteArray()
        self._recording = False
        self._start_time: float = 0.0

        # Spectrum timer (10 Hz while recording)
        self._spectrum_timer = QTimer(self)
        self._spectrum_timer.setInterval(SPECTRUM_INTERVAL_MS)
        self._spectrum_timer.timeout.connect(self._update_spectrum)

        # Cap timer (fires once at max duration)
        self._cap_timer = QTimer(self)
        self._cap_timer.setSingleShot(True)
        self._cap_timer.timeout.connect(self._on_cap_reached)

        # Chunk tracking
        self._next_chunk_sample: int = 0  # sample index for next chunk boundary
        self._chunk_samples: int = chunk_duration_s * SAMPLE_RATE
        self._overlap_samples: int = overlap_duration_s * SAMPLE_RATE

        # Peak tracking for spectrum normalization
        self._spectrum_peak: float = 1.0
        self._spectrum_warmup: int = 0

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def selected_device_id(self) -> str:
        return self._selected_device_id

    @selected_device_id.setter
    def selected_device_id(self, value: str) -> None:
        self._selected_device_id = value

    @staticmethod
    def available_input_devices() -> list[dict[str, str]]:
        """Return list of available audio input devices as {id, description}."""
        devices = []
        for dev in QMediaDevices.audioInputs():
            devices.append(
                {
                    "id": dev.id().data().decode(),
                    "description": dev.description(),
                }
            )
        return devices

    @property
    def full_buffer_int16(self) -> np.ndarray:
        """Return the full recording buffer as int16 numpy array."""
        data = self._buffer.data()
        if not data:
            return np.array([], dtype=np.int16)
        return np.frombuffer(data, dtype=np.int16).copy()

    @property
    def elapsed_seconds(self) -> float:
        if not self._recording:
            return 0.0
        return time.monotonic() - self._start_time

    def toggle(self) -> None:
        if self._recording:
            self.stop()
        else:
            self.start()

    def start(self) -> None:
        if self._recording:
            log.warning("Recording already active, ignoring start()")
            return

        # Configure format
        fmt = QAudioFormat()
        fmt.setSampleRate(SAMPLE_RATE)
        fmt.setChannelCount(CHANNELS)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)

        # Get input device (selected or default)
        device = None
        if self._selected_device_id:
            for dev in QMediaDevices.audioInputs():
                if dev.id().data().decode() == self._selected_device_id:
                    device = dev
                    break
            if device is None:
                log.warning(
                    "Configured device %r not found, falling back to default",
                    self._selected_device_id,
                )
        if device is None:
            device = QMediaDevices.defaultAudioInput()

        if device.isNull():
            log.error("No audio input device available")
            return

        if not device.isFormatSupported(fmt):
            log.error("Default audio device does not support 16kHz mono Int16")
            return

        self._buffer.clear()
        self._next_chunk_sample = self._chunk_samples
        self._spectrum_peak = 1.0
        self._spectrum_warmup = 0
        self._start_time = time.monotonic()

        self._source = QAudioSource(device, fmt, self)
        self._source.setBufferSize(SAMPLE_RATE * SAMPLE_SIZE_BYTES)  # 1s buffer

        self._io_device = self._source.start()
        if self._io_device is None:
            log.error("Failed to start QAudioSource")
            self._source.deleteLater()
            self._source = None
            return

        self._io_device.readyRead.connect(self._on_ready_read)
        self._recording = True
        self._spectrum_timer.start()
        self._cap_timer.start(MAX_RECORDING_SECONDS * 1000)

        set_recording_active(True)
        log.info("Recording started (cap=%ds)", MAX_RECORDING_SECONDS)

    def stop(self) -> None:
        if not self._recording:
            return

        self._recording = False
        self._spectrum_timer.stop()
        self._cap_timer.stop()

        if self._source is not None:
            self._source.stop()
            self._source.deleteLater()
            self._source = None
        self._io_device = None

        # Emit final partial chunk if there's leftover audio
        total_samples = self._buffer.size() // SAMPLE_SIZE_BYTES
        last_emitted = self._next_chunk_sample - self._chunk_samples
        if last_emitted < 0:
            last_emitted = 0
        remaining = total_samples - last_emitted
        if remaining > 0:
            start_sample = max(0, last_emitted - self._overlap_samples)
            start_time = start_sample / SAMPLE_RATE
            chunk_data = self._get_samples(start_sample, total_samples)
            self.chunk_ready.emit(chunk_data, start_time)

        set_recording_active(False)
        self.recording_stopped.emit()
        log.info(
            "Recording stopped (duration=%.1fs, samples=%d)",
            time.monotonic() - self._start_time,
            total_samples,
        )

    def _on_ready_read(self) -> None:
        if self._io_device is None:
            return
        data = self._io_device.readAll()
        if data.isEmpty():
            return
        self._buffer.append(data)
        self._check_chunk_boundary()

    def _check_chunk_boundary(self) -> None:
        total_samples = self._buffer.size() // SAMPLE_SIZE_BYTES
        while total_samples >= self._next_chunk_sample:
            # Emit chunk including overlap from previous
            start_sample = max(
                0, self._next_chunk_sample - self._chunk_samples - self._overlap_samples
            )
            end_sample = self._next_chunk_sample
            start_time = start_sample / SAMPLE_RATE
            chunk_data = self._get_samples(start_sample, end_sample)
            log.info(
                "Emitting chunk: start=%.1fs, end=%.1fs, samples=%d",
                start_time,
                end_sample / SAMPLE_RATE,
                len(chunk_data),
            )
            self.chunk_ready.emit(chunk_data, start_time)
            self._next_chunk_sample += self._chunk_samples
            log.debug(
                "Chunk emitted: start=%.1fs, samples=%d",
                start_time,
                len(chunk_data),
            )

    def _get_samples(self, start: int, end: int) -> np.ndarray:
        byte_start = start * SAMPLE_SIZE_BYTES
        byte_end = end * SAMPLE_SIZE_BYTES
        raw = self._buffer.data()[byte_start:byte_end]
        return np.frombuffer(raw, dtype=np.int16).copy()

    def _update_spectrum(self) -> None:
        total_samples = self._buffer.size() // SAMPLE_SIZE_BYTES
        if total_samples < SPECTRUM_WINDOW_SAMPLES:
            self._overlay_vm.set_bar_levels([0.0] * NUM_BANDS)
            return

        # Take last 50ms of audio
        start = total_samples - SPECTRUM_WINDOW_SAMPLES
        samples = self._get_samples(start, total_samples).astype(np.float32)

        # Noise gate: if RMS is below threshold, show flat bars
        rms = np.sqrt(np.mean(samples * samples))
        if rms < 300.0:
            self._overlay_vm.set_bar_levels([0.0] * NUM_BANDS)
            return

        # Apply Hann window and FFT
        window = np.hanning(len(samples))
        spectrum = np.abs(np.fft.rfft(samples * window))

        # Focus on voice frequencies: 100Hz–4000Hz
        # At 16kHz with 800-sample FFT: bin resolution = 20Hz/bin
        # bin 5 = 100Hz, bin 200 = 4000Hz
        freq_lo = 5
        freq_hi = 200
        voice_spectrum = spectrum[freq_lo:freq_hi]

        # Split into bands using mel-like spacing (logarithmic)
        n_bins = len(voice_spectrum)
        levels = []
        for i in range(NUM_BANDS):
            # Log-spaced band boundaries
            frac_lo = (np.exp(i / NUM_BANDS * np.log(n_bins + 1)) - 1) / n_bins
            frac_hi = (np.exp((i + 1) / NUM_BANDS * np.log(n_bins + 1)) - 1) / n_bins
            b_lo = int(frac_lo * n_bins)
            b_hi = max(b_lo + 1, int(frac_hi * n_bins))
            band_energy = np.mean(voice_spectrum[b_lo:b_hi])
            levels.append(float(band_energy))

        # Normalize with fast-adapting peak (attack instant, decay ~1.5s at 10Hz)
        current_peak = max(levels) if levels else 1.0
        if current_peak > self._spectrum_peak:
            self._spectrum_peak = current_peak
        else:
            self._spectrum_peak *= 0.96  # ~1.5s half-life at 10Hz

        # Floor to avoid division by tiny values
        effective_peak = max(self._spectrum_peak, 1.0)
        levels = [min(1.0, v / effective_peak) for v in levels]

        # Warmup: suppress display for first 300ms to let peak stabilize
        if self._spectrum_warmup < 3:
            self._spectrum_warmup += 1
            return

        self._overlay_vm.set_bar_levels(levels)

    def _on_cap_reached(self) -> None:
        log.warning("Recording cap reached (%ds), stopping", MAX_RECORDING_SECONDS)
        self.stop()
        # Notify user
        try:
            from subprocess import DEVNULL, Popen

            Popen(
                [
                    "notify-send",
                    "VII Recording",
                    f"Recording stopped: {MAX_RECORDING_SECONDS // 3600}h cap reached",
                ],
                start_new_session=True,
                stdout=DEVNULL,
                stderr=DEVNULL,
            )
        except Exception:
            pass
