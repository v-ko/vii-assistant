"""Client for pushing items to curator via HTTP API."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)

CURATOR_SERVER_URL = "http://localhost:3333"


async def push_item(
    feed_name: str,
    content: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Push an item to a curator feed via the HTTP API.

    Args:
        feed_name: The feed name to push to.
        content: The item content dict (keys: text, url, image, video, title).
        metadata: Optional metadata dict (keys: published, etc.).

    Returns:
        dict with status key ("ok" or "error").
    """
    payload: dict[str, Any] = {
        "feed": feed_name,
        "content": content,
    }
    if metadata:
        payload["metadata"] = metadata

    content_summary = {
        k: (f"<{len(v)} chars>" if isinstance(v, str) and len(v) > 200 else v)
        for k, v in content.items()
    }
    log.info(
        "Pushing to curator: feed=%r content=%s metadata=%s",
        feed_name,
        content_summary,
        metadata,
    )

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{CURATOR_SERVER_URL}/push",
                json=payload,
            )
            resp.raise_for_status()
            result = resp.json()
            log.info(
                "Successfully pushed item to feed '%s': %s",
                feed_name,
                json.dumps(result),
            )
            return {"status": "ok"}
    except httpx.HTTPStatusError as exc:
        err_msg = exc.response.text.strip()
        log.error("Curator API error pushing to feed '%s': %s", feed_name, err_msg)
        return {"status": "error", "error_message": err_msg}
    except httpx.ConnectError:
        log.error("Cannot reach curator server at %s", CURATOR_SERVER_URL)
        return {"status": "error", "error_message": "curator server unreachable"}
    except Exception as exc:
        log.error("Failed to push item to curator feed '%s': %s", feed_name, exc)
        return {"status": "error", "error_message": str(exc)}
