"""Transcription client — reuses VII's transcription protocol without Qt deps.

Mirrors the logic from assistant.services.transcription_orchestrator but as a
standalone async client that only depends on httpx and numpy.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import AsyncGenerator

import httpx
import numpy as np

log = logging.getLogger(__name__)

TRANSCRIBE_TIMEOUT_S = 120.0


class TranscriptionClient:
    """Sends audio chunks to the VII inference server /transcribe endpoint.

    Provides the same streaming protocol as TranscriptionOrchestrator:
    feed (int16_audio, start_time) tuples into a queue, get stitched text out.
    """

    def __init__(self, inference_url: str = "http://localhost:8008") -> None:
        self._inference_url = inference_url.rstrip("/")
        self._transcribe_url = f"{self._inference_url}/transcribe"

    async def transcribe_chunk(
        self, client: httpx.AsyncClient, audio_int16: np.ndarray
    ) -> dict:
        """Transcribe a single chunk. Returns {text, words}."""
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        audio_b64 = base64.b64encode(audio_float32.tobytes()).decode("ascii")

        resp = await client.post(
            self._transcribe_url,
            json={
                "model_type": "parakeet-tdt-0.6b-v3-int8",
                "sample_rate": 16000,
                "audio_b64": audio_b64,
            },
        )
        resp.raise_for_status()
        return resp.json()

    async def transcribe_stream(
        self,
        chunk_queue: asyncio.Queue[tuple[np.ndarray, float] | None],
    ) -> AsyncGenerator[str, None]:
        """Consume chunks from queue, transcribe each, yield incremental text.

        Same protocol as TranscriptionOrchestrator.run():
        - Queue receives (audio_int16, chunk_start_time_s) tuples
        - None signals end of recording
        - Yields new text segments as they arrive
        """
        chunk_results: list[tuple[float, str]] = []
        last_yielded_len = 0

        async with httpx.AsyncClient(timeout=TRANSCRIBE_TIMEOUT_S) as client:
            while True:
                item = await chunk_queue.get()
                if item is None:
                    break

                audio_int16, chunk_start_time = item
                log.info(
                    "Transcribing chunk: start=%.1fs, samples=%d",
                    chunk_start_time,
                    len(audio_int16),
                )

                try:
                    data = await self.transcribe_chunk(client, audio_int16)
                except Exception as e:
                    log.error("Chunk transcription failed: %s", e)
                    continue

                text = data.get("text", "")
                log.info("Chunk result: %r", text[:80] if text else "(empty)")
                chunk_results.append((chunk_start_time, text))

                # Simple concatenation (stitching can be improved later by
                # importing stitch_chunk_results from assistant package)
                full_text = " ".join(t for _, t in sorted(chunk_results) if t)
                if len(full_text) > last_yielded_len:
                    new_text = full_text[last_yielded_len:]
                    last_yielded_len = len(full_text)
                    yield new_text
