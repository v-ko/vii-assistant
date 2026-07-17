"""Protocol defining the interface for inference backends."""

from __future__ import annotations

from typing import Any, AsyncIterator, Protocol, runtime_checkable

from PIL import Image


@runtime_checkable
class InferenceBackend(Protocol):
    """Common interface for model backends (HuggingFace, llama.cpp, etc.)."""

    @property
    def state(self) -> str:
        """One of: 'unloaded', 'loading', 'loaded'."""
        ...

    @property
    def current_model_key(self) -> str | None: ...

    async def load_model(self, model_key: str) -> None: ...
    async def unload_model(self) -> None: ...
    def get_state_dict(self) -> dict: ...

    async def generate(
        self,
        messages: list[dict[str, Any]],
        images: list[Image.Image],
        gen_params: dict[str, Any],
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Non-streaming generation.

        Returns dict with keys: status, text, tokens (on success)
        or status, error_message (on failure).
        """
        ...

    def generate_stream(  # Not marked as async because of pylance errors (donno..)
        self,
        messages: list[dict[str, Any]],
        images: list[Image.Image],
        gen_params: dict[str, Any],
        chat_template_kwargs: dict[str, Any] | None = None,
        cancel_event: Any | None = None,
    ) -> AsyncIterator[str]:
        """Streaming generation. Yields text chunks.

        If cancel_event is set, generation should stop early.
        """
        ...
