"""File transcription procedure.

Loads an audio file via QAudioDecoder, chunks it, sends to the inference
server via the existing orchestrator, and copies the result to clipboard.
"""

from __future__ import annotations

import asyncio

import numpy as np
from fusion import get_logger
from fusion.libs.procedure import procedure
from PySide6.QtCore import QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtMultimedia import QAudioDecoder, QAudioFormat

from assistant.facade import vii
from assistant.services.transcription_chunking import (
    CHUNK_DURATION_S,
    OVERLAP_DURATION_S,
    stitch_chunk_results,
)
from assistant.transcription import TranscriptionResult, WordTimestamp

log = get_logger(__name__)

SAMPLE_RATE = 16000


@procedure
async def transcribe_file(file_path: str) -> None:
    """Transcribe an audio file and copy the result to clipboard."""
    import base64

    import httpx

    vs = vii.app.view_state.settings_modal_VS.transcription_section_vs

    vs.sample_file_progress = 0.0

    try:
        # Load and decode audio via QAudioDecoder
        log.info("Loading audio file: %s", file_path)
        audio = await _decode_audio_file(file_path)
        if audio is None or len(audio) == 0:
            log.error("Failed to decode audio file or file is empty")
            vs.sample_file_progress = None
            return

        log.info(
            "Audio loaded: %.1fs, %d samples", len(audio) / SAMPLE_RATE, len(audio)
        )

        # Chunk the audio
        chunk_samples = CHUNK_DURATION_S * SAMPLE_RATE
        overlap_samples = OVERLAP_DURATION_S * SAMPLE_RATE
        step = chunk_samples - overlap_samples

        chunks: list[tuple[np.ndarray, float]] = []
        pos = 0
        while pos < len(audio):
            end = min(pos + chunk_samples, len(audio))
            chunk = audio[pos:end]
            chunk_start_time = pos / SAMPLE_RATE
            chunks.append((chunk, chunk_start_time))
            pos += step

        total_chunks = len(chunks)
        log.info("Split into %d chunks", total_chunks)

        # Transcribe each chunk
        chunk_results: list[tuple[float, TranscriptionResult]] = []

        async with httpx.AsyncClient(timeout=120.0) as client:
            for i, (chunk_audio, chunk_start) in enumerate(chunks):
                # Convert to float32 for server
                audio_f32 = chunk_audio.astype(np.float32) / 32768.0
                audio_b64 = base64.b64encode(audio_f32.tobytes()).decode("ascii")

                log.info(
                    "Transcribing chunk %d/%d (start=%.1fs)",
                    i + 1,
                    total_chunks,
                    chunk_start,
                )
                data = await vii.inference_client.transcribe(
                    audio_b64, sample_rate=SAMPLE_RATE, client=client
                )

                words = [
                    WordTimestamp(word=w["word"], start=w["start"], end=w["end"])
                    for w in data.get("words", [])
                ]
                result = TranscriptionResult(text=data.get("text", ""), words=words)
                chunk_results.append((chunk_start, result))

                # Update progress
                vs.sample_file_progress = (i + 1) / total_chunks

        # Stitch all results
        stitched = stitch_chunk_results(
            chunk_results, chunk_duration_s=CHUNK_DURATION_S
        )
        final_text = stitched.text.strip()

        if final_text:
            QGuiApplication.clipboard().setText(final_text)
            log.info(
                "Transcription complete (%d chars), copied to clipboard",
                len(final_text),
            )
        else:
            log.warning("Transcription produced no text")

    except Exception as exc:
        log.error("File transcription failed: %s", exc, exc_info=True)
    finally:
        vs.sample_file_progress = None


async def _decode_audio_file(file_path: str) -> np.ndarray | None:
    """Decode an audio file to 16kHz mono int16 using QAudioDecoder."""
    loop = asyncio.get_running_loop()
    future: asyncio.Future[np.ndarray | None] = loop.create_future()
    buffers: list[bytes] = []

    fmt = QAudioFormat()
    fmt.setSampleRate(SAMPLE_RATE)
    fmt.setChannelCount(1)
    fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)

    decoder = QAudioDecoder()
    decoder.setAudioFormat(fmt)
    decoder.setSource(QUrl.fromLocalFile(file_path))

    def _on_buffer_ready():
        buf = decoder.read()
        data = buf.constData()
        buffers.append(bytes(data))

    def _on_finished():
        if buffers:
            raw = b"".join(buffers)
            audio = np.frombuffer(raw, dtype=np.int16)
            if not future.done():
                future.set_result(audio)
        else:
            if not future.done():
                future.set_result(None)
        decoder.deleteLater()

    def _on_error(error):
        log.error("QAudioDecoder error: %s", decoder.errorString())
        if not future.done():
            future.set_result(None)
        decoder.deleteLater()

    decoder.bufferReady.connect(_on_buffer_ready)
    decoder.finished.connect(_on_finished)
    decoder.error.connect(_on_error)
    decoder.start()

    return await future
