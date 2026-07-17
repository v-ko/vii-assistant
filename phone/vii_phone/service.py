"""Discord voice bridge — streams voice channel audio to VII transcription.

Connects to a designated Discord voice channel, receives audio from the user,
chunks it into 30s segments (matching TranscriptionOrchestrator protocol),
transcribes via the VII inference server, and on call end sends the full
transcription to the VS Code bridge. Response is TTS'd back into the channel.
"""

from __future__ import annotations

import asyncio
import io
import logging
import struct
import subprocess
import tempfile
import time
from pathlib import Path

import discord
import edge_tts
import httpx
import numpy as np
from vii_phone.transcription_client import TranscriptionClient

log = logging.getLogger(__name__)

# Audio format from Discord: 48kHz stereo s16le (per-user sink)
DISCORD_SAMPLE_RATE = 48000
DISCORD_CHANNELS = 2

# Target format for transcription: 16kHz mono int16
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1

CHUNK_DURATION_S = 30
OVERLAP_DURATION_S = 7

# Silence detection: if no audio for this long, treat as utterance end
SILENCE_TIMEOUT_S = 5.0

# Trigger phrases detected in transcription output.
# "respond" triggers: user finished a thought, wants a response back.
RESPOND_TRIGGERS = [
    "over",
    "send it",
    "go ahead",
    "come back",
    "your turn",
]
# "end call" triggers: user is done, disconnect.
END_CALL_TRIGGERS = [
    "over and out",
    "out",
    "disconnect",
    "end call",
    "signing off",
]


class ViiPhoneService:
    """Discord voice bridge to VII transcription + VS Code chat."""

    def __init__(
        self,
        discord_token: str,
        channel_id: int,
        inference_url: str = "http://localhost:8008",
        vscode_bridge_url: str = "http://localhost:51178",
    ) -> None:
        self._token = discord_token
        self._channel_id = channel_id
        self._inference_url = inference_url
        self._vscode_bridge_url = vscode_bridge_url

        self._transcription_client = TranscriptionClient(inference_url)

        # Discord bot setup
        intents = discord.Intents.default()
        intents.voice_states = True
        intents.message_content = True
        self._bot = discord.Bot(intents=intents)
        self._voice_client: discord.VoiceClient | None = None

        # Audio buffering state
        self._audio_buffer = bytearray()
        self._buffer_lock = asyncio.Lock()
        self._chunk_queue: asyncio.Queue[tuple[np.ndarray, float] | None] = (
            asyncio.Queue()
        )
        self._recording_start_time: float = 0.0
        self._last_audio_time: float = 0.0
        self._is_recording = False
        self._chunk_task: asyncio.Task | None = None
        self._silence_task: asyncio.Task | None = None
        self._transcription_task: asyncio.Task | None = None

        # Text channel for fallback text responses
        self._text_channel: discord.TextChannel | None = None

        self._setup_events()

    def _setup_events(self) -> None:
        @self._bot.event
        async def on_ready():
            log.info("vii-phone bot connected as %s", self._bot.user)
            channel = self._bot.get_channel(self._channel_id)
            if channel is None:
                log.error("Channel %d not found", self._channel_id)
                return
            if not isinstance(channel, discord.VoiceChannel):
                log.error("Channel %d is not a voice channel", self._channel_id)
                return
            log.info("Waiting for user in voice channel: %s", channel.name)

        @self._bot.event
        async def on_voice_state_update(
            member: discord.Member,
            before: discord.VoiceState,
            after: discord.VoiceState,
        ):
            # Ignore bot's own state changes
            if member == self._bot.user:
                return

            # User joined our channel
            if (
                after.channel
                and after.channel.id == self._channel_id
                and (before.channel is None or before.channel.id != self._channel_id)
            ):
                log.info("User %s joined voice channel", member.display_name)
                await self._join_and_start(after.channel, member)

            # User left our channel
            elif (
                before.channel
                and before.channel.id == self._channel_id
                and (after.channel is None or after.channel.id != self._channel_id)
            ):
                log.info("User %s left voice channel", member.display_name)
                await self._stop_and_process()

    async def _join_and_start(
        self, channel: discord.VoiceChannel, member: discord.Member
    ) -> None:
        """Join the voice channel and start receiving audio."""
        if self._voice_client and self._voice_client.is_connected():
            return

        self._voice_client = await channel.connect()
        self._is_recording = True
        self._recording_start_time = time.monotonic()
        self._last_audio_time = time.monotonic()
        self._audio_buffer = bytearray()

        # Find a text channel in the same guild for responses
        guild = channel.guild
        for tc in guild.text_channels:
            if tc.permissions_for(guild.me).send_messages:
                self._text_channel = tc
                break

        # Start receiving audio
        self._voice_client.start_recording(
            ViiAudioSink(self._on_audio_data),
            self._on_recording_stopped,
            channel,
        )

        # Start chunking and transcription tasks
        self._chunk_task = asyncio.create_task(self._chunk_loop())
        self._transcription_task = asyncio.create_task(self._transcription_loop())
        self._silence_task = asyncio.create_task(self._silence_monitor())

        log.info("Recording started")

    async def _stop_and_process(self) -> None:
        """Stop recording, finalize transcription, send to VS Code, TTS response."""
        if not self._is_recording:
            return
        self._is_recording = False

        # Stop discord recording
        if self._voice_client and self._voice_client.is_connected():
            self._voice_client.stop_recording()

        # Flush remaining buffer as final chunk
        await self._flush_buffer()

        # Signal end of recording
        await self._chunk_queue.put(None)

        # Cancel silence monitor
        if self._silence_task:
            self._silence_task.cancel()

        # Wait for transcription to finish
        if self._transcription_task:
            try:
                full_text = await self._transcription_task
            except Exception as e:
                log.error("Transcription failed: %s", e)
                full_text = ""
        else:
            full_text = ""

        log.info("Full transcription: %s", full_text[:200])

        if full_text.strip():
            # Send to VS Code bridge and get response
            response = await self._send_to_vscode(full_text)
            log.info("VS Code response: %s", response[:200] if response else "(empty)")

            # Send text response in channel
            if self._text_channel and response:
                # Discord message limit is 2000 chars
                for i in range(0, len(response), 2000):
                    await self._text_channel.send(response[i : i + 2000])

            # TTS the response back into voice channel
            if response and self._voice_client and self._voice_client.is_connected():
                await self._tts_and_play(response)

        # Disconnect from voice
        if self._voice_client and self._voice_client.is_connected():
            await self._voice_client.disconnect()
        self._voice_client = None

        # Cleanup tasks
        if self._chunk_task and not self._chunk_task.done():
            self._chunk_task.cancel()

    def _on_audio_data(self, user_id: int, data: bytes) -> None:
        """Callback from ViiAudioSink — raw PCM s16le 48kHz stereo."""
        self._last_audio_time = time.monotonic()
        self._audio_buffer.extend(data)

    def _on_recording_stopped(self, sink, *args) -> None:
        """Called when discord stops recording."""
        log.debug("Discord recording stopped callback")

    async def _chunk_loop(self) -> None:
        """Periodically slice the audio buffer into transcription-sized chunks."""
        chunk_samples = CHUNK_DURATION_S * TARGET_SAMPLE_RATE
        overlap_samples = OVERLAP_DURATION_S * TARGET_SAMPLE_RATE
        step_samples = chunk_samples - overlap_samples

        # Bytes per step in source format (48kHz stereo s16le)
        src_bytes_per_sample = DISCORD_CHANNELS * 2  # 2 bytes per sample per channel
        # How many source bytes correspond to one step worth of target samples
        src_step_bytes = (
            int(step_samples * (DISCORD_SAMPLE_RATE / TARGET_SAMPLE_RATE))
            * src_bytes_per_sample
        )

        next_chunk_offset = 0
        src_chunk_bytes = (
            int(chunk_samples * (DISCORD_SAMPLE_RATE / TARGET_SAMPLE_RATE))
            * src_bytes_per_sample
        )

        while self._is_recording:
            await asyncio.sleep(1.0)  # Check every second

            buffer_len = len(self._audio_buffer)
            if buffer_len >= next_chunk_offset + src_chunk_bytes:
                # Extract chunk from buffer
                raw = bytes(
                    self._audio_buffer[
                        next_chunk_offset : next_chunk_offset + src_chunk_bytes
                    ]
                )
                chunk_start_time = (
                    next_chunk_offset / src_bytes_per_sample
                ) / DISCORD_SAMPLE_RATE

                # Convert 48kHz stereo → 16kHz mono int16
                audio_int16 = self._resample_to_16k_mono(raw)

                await self._chunk_queue.put((audio_int16, chunk_start_time))
                next_chunk_offset += src_step_bytes

    async def _flush_buffer(self) -> None:
        """Flush any remaining audio in the buffer as a final chunk."""
        if not self._audio_buffer:
            return

        src_bytes_per_sample = DISCORD_CHANNELS * 2
        # Convert whatever is left
        raw = bytes(self._audio_buffer)
        chunk_start_time = 0.0  # Simplified for final flush

        audio_int16 = self._resample_to_16k_mono(raw)
        if len(audio_int16) > TARGET_SAMPLE_RATE:  # At least 1s of audio
            await self._chunk_queue.put((audio_int16, chunk_start_time))

    def _resample_to_16k_mono(self, raw_48k_stereo: bytes) -> np.ndarray:
        """Convert 48kHz stereo s16le bytes to 16kHz mono int16 numpy array."""
        # Parse as int16 stereo
        samples = np.frombuffer(raw_48k_stereo, dtype=np.int16)
        # Stereo to mono: average channels
        stereo = samples.reshape(-1, DISCORD_CHANNELS)
        mono = stereo.mean(axis=1).astype(np.int16)
        # Downsample 48kHz → 16kHz (factor of 3)
        downsampled = mono[::3]
        return downsampled

    async def _transcription_loop(self) -> str:
        """Consume chunks from queue, transcribe, stitch. Returns full text."""
        full_text = ""
        async for segment in self._transcription_client.transcribe_stream(
            self._chunk_queue
        ):
            full_text += segment
            log.info("Transcription progress: ...%s", full_text[-80:])
        return full_text

    async def _silence_monitor(self) -> None:
        """Detect silence and trigger end of call."""
        while self._is_recording:
            await asyncio.sleep(0.5)
            elapsed = time.monotonic() - self._last_audio_time
            if elapsed > SILENCE_TIMEOUT_S and self._is_recording:
                log.info("Silence detected (%.1fs), ending call", elapsed)
                await self._stop_and_process()
                return

    async def _send_to_vscode(self, text: str) -> str:
        """Send transcription to VS Code bridge extension, return response."""
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{self._vscode_bridge_url}/chat",
                    json={"message": text},
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("response", "")
        except Exception as e:
            log.error("Failed to reach VS Code bridge: %s", e)
            return f"[Error: could not reach VS Code bridge: {e}]"

    async def _tts_and_play(self, text: str) -> None:
        """Convert text to speech and play in voice channel."""
        if not self._voice_client or not self._voice_client.is_connected():
            return

        try:
            # Generate TTS audio with edge-tts
            communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        f.write(chunk["data"])

            # Play via ffmpeg audio source
            source = discord.FFmpegPCMAudio(tmp_path)
            self._voice_client.play(source)

            # Wait for playback to finish
            while self._voice_client.is_playing():
                await asyncio.sleep(0.5)

            # Cleanup
            Path(tmp_path).unlink(missing_ok=True)

        except Exception as e:
            log.error("TTS playback failed: %s", e)

    async def start(self) -> None:
        """Start the Discord bot (blocking)."""
        await self._bot.start(self._token)

    def run(self) -> None:
        """Start the Discord bot using its own event loop (recommended)."""
        self._bot.run(self._token)

    async def stop(self) -> None:
        """Gracefully shut down."""
        if self._voice_client and self._voice_client.is_connected():
            await self._voice_client.disconnect()
        await self._bot.close()


class ViiAudioSink(discord.sinks.Sink):
    """Custom audio sink that forwards raw PCM to a callback."""

    def __init__(self, on_data: callable):
        super().__init__()
        self._on_data = on_data

    @discord.sinks.Filters.container
    def write(self, data, user) -> None:
        """Called by py-cord with decoded PCM for each user."""
        from discord.voice.packets import VoiceData

        pcm = data.pcm if isinstance(data, VoiceData) else data
        user_id = getattr(user, "id", user) if user else 0
        self._on_data(user_id, pcm)

    def cleanup(self) -> None:
        self.finished = True
