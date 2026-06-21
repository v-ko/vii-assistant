"""Transcription orchestrator — client-side chunking, server inference, stitching.

Exposes an async generator that yields incrementally stitched text segments.
The orchestrator tracks in-flight requests and stale session detection via
a generation counter.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import AsyncGenerator, Callable

import httpx
import numpy as np

from assistant.services.inference_client import InferenceServerClient
from assistant.services.transcription_chunking import (
    CHUNK_DURATION_S,
    stitch_chunk_results,
)
from assistant.transcription_types import TranscriptionResult, WordTimestamp

log = logging.getLogger(__name__)

TRANSCRIBE_TIMEOUT_S = 120.0  # per-chunk timeout


class TranscriptionOrchestrator:
    """Client-side orchestrator: sends chunks to server, stitches results.

    Usage:
        orch = TranscriptionOrchestrator(inference_client)
        async for segment in orch.run(recording_service):
            # segment is the latest stitched text (incremental)
            ...
    """

    def __init__(self, inference_client: InferenceServerClient) -> None:
        self._inference_client = inference_client
        self._on_transcribing_changed: Callable[[bool], None] | None = None
        self._generation: int = 0
        self._in_flight: int = 0
        self.last_result: TranscriptionResult | None = None

    def set_on_transcribing_changed(self, callback: Callable[[bool], None]) -> None:
        """Wire a callback to be notified when transcription starts/stops."""
        self._on_transcribing_changed = callback

    async def run(
        self,
        chunk_queue: asyncio.Queue[tuple[np.ndarray, float] | None],
        *,
        generation: int | None = None,
    ) -> AsyncGenerator[str, None]:
        """Consume chunks from queue, transcribe each, yield stitched text.

        The queue should receive (audio_int16, chunk_start_time_s) tuples.
        Send None to signal end of recording.

        Yields incremental stitched text after each chunk result.
        """
        self._generation += 1
        gen = generation if generation is not None else self._generation
        self._in_flight = 0

        chunk_results: list[tuple[float, TranscriptionResult]] = []
        last_yielded_text = ""
        first_chunk = True

        log.info("Orchestrator started (generation=%d)", gen)

        try:
            async with httpx.AsyncClient(timeout=TRANSCRIBE_TIMEOUT_S) as client:
                while True:
                    log.debug("Waiting for chunk from queue...")
                    item = await chunk_queue.get()

                    # Check for session staleness
                    if self._generation != gen:
                        log.info("Session generation stale, stopping orchestrator")
                        return

                    if item is None:
                        # End of recording
                        log.info("Received end-of-recording sentinel")
                        break

                    if first_chunk:
                        if self._on_transcribing_changed:
                            self._on_transcribing_changed(True)
                        first_chunk = False

                    audio_int16, chunk_start_time = item
                    self._in_flight += 1
                    log.info(
                        "Processing chunk: start=%.1fs, samples=%d, sending to server",
                        chunk_start_time,
                        len(audio_int16),
                    )

                    try:
                        result = await self._transcribe_chunk(
                            client, audio_int16, chunk_start_time
                        )
                    except Exception as exc:
                        self._in_flight -= 1
                        log.error(
                            "Chunk transcription failed (start=%.1fs): %s",
                            chunk_start_time,
                            exc,
                        )
                        raise

                    self._in_flight -= 1
                    log.info(
                        "Chunk transcribed: start=%.1fs, text=%r",
                        chunk_start_time,
                        result.text[:80] if result.text else "(empty)",
                    )
                    chunk_results.append((chunk_start_time, result))

                    # Stitch all results so far
                    stitched = stitch_chunk_results(
                        chunk_results, chunk_duration_s=CHUNK_DURATION_S
                    )
                    self.last_result = stitched
                    if stitched.text != last_yielded_text:
                        # Yield only the new portion
                        new_text = stitched.text[len(last_yielded_text) :]
                        if new_text.startswith(" "):
                            new_text = new_text  # preserve leading space
                        last_yielded_text = stitched.text
                        yield new_text
        finally:
            if self._on_transcribing_changed:
                self._on_transcribing_changed(False)

    async def _transcribe_chunk(
        self,
        client: httpx.AsyncClient,
        audio_int16: np.ndarray,
        chunk_start_time: float,
    ) -> TranscriptionResult:
        """Send a single chunk to the inference server and parse the result."""
        # Convert int16 to float32 for server endpoint
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        audio_b64 = base64.b64encode(audio_float32.tobytes()).decode("ascii")

        log.debug(
            "Sending chunk to server (start=%.1fs, samples=%d)",
            chunk_start_time,
            len(audio_int16),
        )
        data = await self._inference_client.transcribe(
            audio_b64, sample_rate=16000, client=client
        )

        words = [
            WordTimestamp(word=w["word"], start=w["start"], end=w["end"])
            for w in data.get("words", [])
        ]
        text = data.get("text", "")
        return TranscriptionResult(text=text, words=words)

    def cancel(self) -> None:
        """Bump generation to invalidate any in-flight session."""
        self._generation += 1
        self._in_flight = 0
