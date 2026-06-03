"""HTTP client for the VII inference server."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from typing import Any, Callable

import httpx
from sivkit import get_logger

from assistant.constants import INFERENCE_HTTP_BASE

log = get_logger(__name__)


class InferenceServerClient:
    """Thin HTTP client for the inference server.

    Owns the base URL and provides methods for all REST interactions.
    """

    def __init__(self, base_url: str = INFERENCE_HTTP_BASE) -> None:
        self.base_url = base_url

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

    # ── Health / model management ────────────────────────────────

    def check_status_bg(self, callback: Callable[[bool, str, str], None]) -> None:
        """Run health check in background thread."""
        threading.Thread(
            target=self._do_health_check, args=(callback,), daemon=True
        ).start()

    def _do_health_check(self, callback: Callable[[bool, str, str], None]) -> None:
        url = f"{self.base_url}/status"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())
            model_info = data.get("model", {})
            model_state = model_info.get("state", "unknown")
            model_key = model_info.get("model_key") or ""
            callback(True, model_state, model_key)
        except Exception:
            callback(False, "unknown", "")

    def load_model_bg(
        self, model_key: str, callback: Callable[[bool, str, str], None]
    ) -> None:
        """Load/unload model in background thread."""
        threading.Thread(
            target=self._do_model_load, args=(model_key, callback), daemon=True
        ).start()

    def _do_model_load(
        self, model_key: str, callback: Callable[[bool, str, str], None]
    ) -> None:
        url = f"{self.base_url}/model"
        if model_key == "none":
            req = urllib.request.Request(url, method="DELETE")
        else:
            payload = json.dumps({"model_key": model_key}).encode()
            req = urllib.request.Request(
                url,
                data=payload,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            model_info = data.get("model", {})
            model_state = model_info.get("state", "unknown")
            mk = model_info.get("model_key") or ""
            callback(True, model_state, mk)
        except Exception:
            callback(False, "unknown", "")

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
            return resp.json()

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
