"""HTTP client for the VII inference server."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Callable

import httpx
import sivkit

from assistant.constants import INFERENCE_HTTP_BASE

if TYPE_CHECKING:
    from assistant.view_states.terminal import TerminalViewState

log = logging.getLogger(__name__)


class InferenceServerClient:
    """Thin HTTP client for the inference server.

    Owns the base URL and the live server status (connection + model state),
    exposed as plain attributes. A service-owned poll loop keeps the status
    fresh; a wired callback notifies a projector on every change.
    """

    def __init__(
        self,
        base_url: str = INFERENCE_HTTP_BASE,
        *,
        poll_interval: float = 3.0,
    ) -> None:
        self.base_url = base_url
        self.poll_interval = poll_interval
        # Live status (source of truth — projected to the view state).
        self.connected: bool = False
        self.model_state: str = "unknown"
        self.model_key: str = ""
        self._on_status_changed: Callable[[], None] | None = None
        # Optional GUI dependency: when set, polling is skipped while the
        # terminal window is hidden (avoids spamming /status for the regular
        # app). Headless runs leave this None and always poll.
        self._terminal_state: TerminalViewState | None = None
        self._polling: bool = False
        # Monotonic timestamp of the last successful inference — a recent one
        # already proves the server is live, so we skip the redundant poll.
        self._last_inference_at: float = float("-inf")

    @property
    def ws_url(self) -> str:
        return (
            self.base_url.replace("http://", "ws://", 1).replace(
                "https://", "wss://", 1
            )
            + "/ws/context"
        )

    @property
    def host_display(self) -> str:
        return self.base_url.split("://", 1)[-1]

    # ── Status (connection + model) ──────────────────────────────

    def set_on_status_changed(self, callback: Callable[[], None]) -> None:
        """Wire a callback fired on the main thread after each status update."""
        self._on_status_changed = callback

    def _set_status(self, connected: bool, model_state: str, model_key: str) -> None:
        self.connected = connected
        self.model_state = model_state
        self.model_key = model_key
        if self._on_status_changed is not None:
            self._on_status_changed()

    async def check_status(self) -> None:
        """Fetch /status and update the live status attributes."""
        url = f"{self.base_url}/status"
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
            model_info = data.get("model", {})
            self._set_status(
                True,
                model_info.get("state", "unknown"),
                model_info.get("model_key") or "",
            )
        except Exception:
            self._set_status(False, "unknown", "")

    async def load_model(self, model_key: str) -> None:
        """Load (POST) or unload (DELETE) a model and update the status."""
        url = f"{self.base_url}/model"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                if model_key == "none":
                    resp = await client.delete(url)
                else:
                    resp = await client.post(url, json={"model_key": model_key})
                resp.raise_for_status()
                data = resp.json()
            model_info = data.get("model", {})
            self._set_status(
                True,
                model_info.get("state", "unknown"),
                model_info.get("model_key") or "",
            )
        except Exception:
            self._set_status(False, "unknown", "")

    def start_status_polling(self, interval: float | None = None) -> None:
        """Start the service-owned poll loop (idempotent).

        Deferred onto the main loop so it is safe to call during init,
        before the asyncio loop is running. Defaults to ``poll_interval``.
        """
        if self._polling:
            return
        self._polling = True
        interval = self.poll_interval if interval is None else interval
        sivkit.call_delayed(lambda: asyncio.ensure_future(self._poll_loop(interval)), 0)

    def set_terminal_state(self, terminal_state: "TerminalViewState") -> None:
        """Inject the GUI terminal state so polling can pause when it's hidden."""
        self._terminal_state = terminal_state

    async def _poll_loop(self, interval: float) -> None:
        while True:
            terminal_open = self._terminal_state is None or self._terminal_state.visible
            recent_inference = time.monotonic() - self._last_inference_at < interval
            if terminal_open and not recent_inference:
                await self.check_status()
            await asyncio.sleep(interval)

    # ── Context debug ────────────────────────────────────────────

    async def get_context_debug(self, focus_mode: str) -> str:
        """Fetch context debug info and format it for display.

        The server returns structured messages with image metadata.
        This method formats them into readable text, reducing images to
        summary lines.
        """
        url = f"{self.base_url}/context/{focus_mode}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        if data.get("status") == "error":
            return data.get("error_message", "Unknown error")

        prompt = data.get("prompt", "")
        messages = data.get("messages", [])
        image_count = data.get("image_count", 0)

        # Format messages for display, replacing images with summaries
        parts: list[str] = []
        if prompt:
            parts.append(prompt)
        elif messages:
            parts.append(self._format_messages(messages))

        return "\n".join(parts) if parts else "(empty context)"

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        """Format structured messages into readable debug text."""
        lines: list[str] = []
        for msg in messages:
            role = msg.get("role", "?")
            lines.append(f"[{role}]")
            for part in msg.get("content", []):
                if part.get("type") == "text":
                    lines.append(part.get("text", ""))
                elif part.get("type") == "image":
                    w = part.get("width", "?")
                    h = part.get("height", "?")
                    lines.append(f"  [image {w}x{h}]")
            lines.append("")
        return "\n".join(lines)

    # ── Inference ─────────────────────────────────────────────────

    async def infer(
        self,
        context_data: list[dict[str, Any]],
        generation_params: dict[str, Any] | None = None,
        *,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """POST to /infer with context_data and optional generation_params.

        Returns the parsed JSON response dict.
        Raises on HTTP errors.
        """
        body: dict[str, Any] = {"context_data": context_data}
        if generation_params:
            body["generation_params"] = generation_params

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{self.base_url}/infer", json=body)
            resp.raise_for_status()
            result = resp.json()
        self._last_inference_at = time.monotonic()
        return result

    # ── Transcription ────────────────────────────────────────────

    @property
    def transcribe_url(self) -> str:
        return f"{self.base_url}/transcribe"

    async def transcribe(
        self,
        audio_b64: str,
        sample_rate: int = 16000,
        model_type: str = "parakeet-tdt-0.6b-v3-int8",
        *,
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> dict[str, Any]:
        """POST to /transcribe with base64-encoded audio.

        If *client* is provided it is reused (useful for batch/streaming
        scenarios); otherwise a fresh client is created per call.

        Returns the parsed JSON response dict.
        Raises on HTTP errors or server-reported errors.
        """
        payload = {
            "model_type": model_type,
            "sample_rate": sample_rate,
            "audio_b64": audio_b64,
        }

        if client is not None:
            resp = await client.post(self.transcribe_url, json=payload)
        else:
            async with httpx.AsyncClient(timeout=timeout) as c:
                resp = await c.post(self.transcribe_url, json=payload)

        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "success":
            error_msg = data.get("error_message", "Unknown server error")
            raise RuntimeError(f"Transcription server error: {error_msg}")

        return data
