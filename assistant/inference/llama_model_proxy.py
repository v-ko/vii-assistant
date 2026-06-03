"""LlamaModelProxy: manages a llama-server subprocess and proxies inference."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import subprocess
import threading
from collections.abc import AsyncIterator
from typing import Any

import httpx
from PIL import Image

from assistant.model_configs import MODEL_SPECS

log = logging.getLogger(__name__)

# Timeout for waiting for llama-server to become healthy after start
_HEALTH_POLL_INTERVAL = 1.0  # seconds
_HEALTH_POLL_TIMEOUT = 600.0  # seconds (model download + load can be slow)


def _messages_with_base64_images(
    messages: list[dict[str, Any]], images: list[Image.Image]
) -> list[dict[str, Any]]:
    """Replace {"type": "image"} placeholders with base64 image_url entries."""
    img_iter = iter(images)
    result: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            result.append(msg)
            continue
        new_content: list[dict[str, Any]] = []
        for part in content:
            if part.get("type") == "image":
                img = next(img_iter)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                new_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )
            else:
                new_content.append(part)
        result.append({**msg, "content": new_content})
    return result


def _map_gen_params(gen_params: dict[str, Any]) -> dict[str, Any]:
    """Map internal gen_params to OpenAI API parameters."""
    api_params: dict[str, Any] = {}
    if "max_new_tokens" in gen_params:
        api_params["max_tokens"] = gen_params["max_new_tokens"]
    if "temperature" in gen_params:
        temp = float(gen_params["temperature"])
        api_params["temperature"] = temp
    if gen_params.get("do_sample") is False and "temperature" not in api_params:
        api_params["temperature"] = 0.0
    if "top_p" in gen_params:
        api_params["top_p"] = gen_params["top_p"]
    if "top_k" in gen_params:
        api_params["top_k"] = gen_params["top_k"]
    if "repetition_penalty" in gen_params:
        api_params["repeat_penalty"] = gen_params["repetition_penalty"]
    return api_params


class LlamaModelProxy:
    """Manages a llama-server subprocess and proxies inference requests.

    Implements the InferenceBackend protocol.
    """

    def __init__(self, port: int = 8081, host: str = "127.0.0.1"):
        self._port = port
        self._host = host
        self.base_url = f"http://{host}:{port}"
        self._process: subprocess.Popen[bytes] | None = None
        self._model_key: str | None = None
        self._state: str = "unloaded"
        self._load_lock = asyncio.Lock()

    @property
    def state(self) -> str:
        return self._state

    @property
    def current_model_key(self) -> str | None:
        return self._model_key

    def get_state_dict(self) -> dict:
        return {
            "model_key": self._model_key,
            "state": self._state,
        }

    async def health_check(self) -> bool:
        """Check if llama-server is healthy and ready."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/health")
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def load_model(self, model_key: str) -> None:
        """Start llama-server with the given model config."""
        if model_key not in MODEL_SPECS:
            raise ValueError(f"Unknown model key: {model_key}")

        spec = MODEL_SPECS[model_key]
        if spec.get("backend") != "llama_cpp":
            raise ValueError(f"Model {model_key} is not a llama_cpp backend model")

        async with self._load_lock:
            if self._model_key == model_key and self._state == "loaded":
                # Check if still healthy
                if await self.health_check():
                    log.info(
                        "LlamaModelProxy: %s already loaded and healthy", model_key
                    )
                    return

            self._state = "loading"
            log.info("LlamaModelProxy: loading model %s ...", model_key)

            # Kill existing process if any
            await self._kill_process()

            # Build command
            model_id = spec["id"]
            extra_args = spec.get("llama_cpp_args", [])
            cmd = [
                "llama-server",
                "-hf",
                model_id,
                "--host",
                self._host,
                "--port",
                str(self._port),
                "--reasoning",
                "off",
                *extra_args,
            ]

            # Log the resolved model file for traceability
            hff_file = None
            for i, arg in enumerate(extra_args):
                if arg == "-hff" and i + 1 < len(extra_args):
                    hff_file = extra_args[i + 1]
                    break
            log.info(
                "LlamaModelProxy: loading %s (file: %s)",
                model_key,
                hff_file or "auto-selected",
            )
            log.info("LlamaModelProxy: starting: %s", " ".join(cmd))

            try:
                self._process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )

                # Wait for health
                healthy = await self._wait_for_health()
                if not healthy:
                    await self._kill_process()
                    self._state = "unloaded"
                    self._model_key = None
                    raise RuntimeError(
                        f"llama-server failed to become healthy within "
                        f"{_HEALTH_POLL_TIMEOUT}s for model {model_key}"
                    )

                self._model_key = model_key
                self._state = "loaded"
                log.info("LlamaModelProxy: %s loaded successfully", model_key)
            except Exception:
                self._state = "unloaded"
                self._model_key = None
                await self._kill_process()
                raise

    async def unload_model(self) -> None:
        """Terminate llama-server process."""
        await self._kill_process()
        self._state = "unloaded"
        self._model_key = None
        log.info("LlamaModelProxy: model unloaded")

    async def generate(
        self,
        messages: list[dict[str, Any]],
        images: list[Image.Image],
        gen_params: dict[str, Any],
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Non-streaming generation via /v1/chat/completions."""
        if self._state != "loaded":
            return {"status": "error", "error_message": "No model loaded"}

        api_messages = _messages_with_base64_images(messages, images)
        api_params = _map_gen_params(gen_params)

        body: dict[str, Any] = {
            "messages": api_messages,
            "stream": False,
            **api_params,
        }
        if chat_template_kwargs:
            body["chat_template_kwargs"] = chat_template_kwargs

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=body,
                )
                resp.raise_for_status()
                data = resp.json()

            choice = data["choices"][0]
            text = choice["message"]["content"].strip()
            usage = data.get("usage", {})
            tokens = usage.get("completion_tokens")
            log.info(
                "LlamaModelProxy: generate complete text_len=%d tokens=%s",
                len(text),
                tokens,
            )
            return {"status": "success", "text": text, "tokens": tokens}
        except Exception as exc:
            log.error("LlamaModelProxy: generate failed: %s", exc, exc_info=True)
            return {"status": "error", "error_message": str(exc)}

    async def generate_stream(
        self,
        messages: list[dict[str, Any]],
        images: list[Image.Image],
        gen_params: dict[str, Any],
        chat_template_kwargs: dict[str, Any] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> AsyncIterator[str]:
        """Streaming generation via SSE from /v1/chat/completions."""
        if self._state != "loaded":
            return

        api_messages = _messages_with_base64_images(messages, images)
        api_params = _map_gen_params(gen_params)

        body: dict[str, Any] = {
            "messages": api_messages,
            "stream": True,
            **api_params,
        }
        if chat_template_kwargs:
            body["chat_template_kwargs"] = chat_template_kwargs

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    json=body,
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if cancel_event and cancel_event.is_set():
                            break
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:]  # strip "data: "
                        if payload.strip() == "[DONE]":
                            break
                        try:
                            import json

                            chunk_data = json.loads(payload)
                            delta = chunk_data["choices"][0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                yield content
                        except (KeyError, json.JSONDecodeError):
                            continue
        except Exception as exc:
            log.error("LlamaModelProxy: generate_stream failed: %s", exc, exc_info=True)

    # --- Internal helpers ---

    async def _wait_for_health(self) -> bool:
        """Poll /health until ready or timeout."""
        elapsed = 0.0
        while elapsed < _HEALTH_POLL_TIMEOUT:
            # Check if process died
            if self._process and self._process.poll() is not None:
                rc = self._process.returncode
                # Capture whatever the process printed before dying
                output = ""
                if self._process.stdout:
                    output = self._process.stdout.read().decode(errors="replace")
                log.error(
                    "LlamaModelProxy: process exited with code %d\n%s",
                    rc,
                    output[-2000:] if output else "(no output)",
                )
                return False
            if await self.health_check():
                return True
            await asyncio.sleep(_HEALTH_POLL_INTERVAL)
            elapsed += _HEALTH_POLL_INTERVAL
        return False

    async def _kill_process(self) -> None:
        """Terminate the llama-server subprocess if running."""
        if self._process is None:
            return
        if self._process.poll() is None:
            log.info(
                "LlamaModelProxy: terminating llama-server (pid=%d)", self._process.pid
            )
            self._process.terminate()
            try:
                await asyncio.to_thread(self._process.wait, timeout=10)
            except subprocess.TimeoutExpired:
                log.warning("LlamaModelProxy: killing llama-server forcefully")
                self._process.kill()
                await asyncio.to_thread(self._process.wait)
        self._process = None
