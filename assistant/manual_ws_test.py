"""Manual websocket test: send prompt + image + request and print responses.

Usage:
  python -m assistant.scripts.manual_ws_test --url ws://desk_local:8000/ws/context
"""

from __future__ import annotations

import asyncio
import base64
import json

import numpy as np
import websockets
from fusion.libs.entity import load_from_dict
from fusion.libs.entity.change import Change
from PIL import Image

# --- Configuration (hard-coded) ---
STREAM = True  # set False to test non-streaming
WAIT_SECONDS = 10.0
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.0


def build_gradient_b64(size: int = 224) -> str:
    data = np.linspace(0, 255, num=size * size * 3, dtype=np.uint8).reshape(
        size, size, 3
    )
    image = Image.fromarray(data, mode="RGB")
    from io import BytesIO

    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


async def main() -> None:
    async with websockets.connect("ws://localhost:8000/ws/context") as ws:
        print("Connected. Waiting for initial messages...")

        async def recv_loop():
            try:
                async for raw in ws:
                    try:
                        payload = json.loads(raw)
                    except Exception:
                        print("<-", raw)
                        continue
                    mtype = payload.get("type") if isinstance(payload, dict) else None
                    if mtype == "AppendItemContentText":
                        print(
                            f"<- STREAM {payload['payload']['item_id']}:"
                            f" {payload['payload']['text']!r}"
                        )
                    else:
                        print("<-", json.dumps(payload)[:2000])
            except Exception as exc:  # noqa: BLE001
                print("Receive loop ended:", exc)

        recv_task = asyncio.create_task(recv_loop())

        # 1) Prompt item
        prompt_item = {
            "type_name": "ContextItem",
            "id": "prompt-1",
            "position": 0,
            "size": [0, 0],
            "content": {"text": "Describe the dominant colors in this image."},
            "request": None,
            "metadata": None,
        }
        if "type_name" not in prompt_item:
            prompt_item["type_name"] = "ContextItem"
        prompt_entity = load_from_dict(prompt_item)
        await ws.send(
            json.dumps(
                {
                    "type": "Change",
                    "payload": Change.CREATE(prompt_entity).as_safe_delta_dict(),
                }
            )
        )

        # 2) Image item
        image_b64 = build_gradient_b64(256)
        image_item = {
            "type_name": "ContextItem",
            "id": "image-1",
            "position": 1,
            "size": [0, 0],
            "content": {"image": image_b64},
            "request": None,
            "metadata": None,
        }
        if "type_name" not in image_item:
            image_item["type_name"] = "ContextItem"
        image_entity = load_from_dict(image_item)
        await ws.send(
            json.dumps(
                {
                    "type": "Change",
                    "payload": Change.CREATE(image_entity).as_safe_delta_dict(),
                }
            )
        )

        # 3) Request item
        request_item = {
            "type_name": "ContextItem",
            "id": "request-1",
            "position": 2,
            "size": [0, 0],
            "content": {"text": "Please answer now."},
            "request": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": TEMPERATURE,
                "stream": STREAM,
            },
            "metadata": None,
        }
        if "type_name" not in request_item:
            request_item["type_name"] = "ContextItem"
        request_entity = load_from_dict(request_item)
        await ws.send(
            json.dumps(
                {
                    "type": "Change",
                    "payload": Change.CREATE(request_entity).as_safe_delta_dict(),
                }
            )
        )

        print("Sent prompt, image and request. Waiting for responses...")
        await asyncio.sleep(WAIT_SECONDS)
        recv_task.cancel()
        try:
            await ws.close()
        except Exception:
            pass


"""No CLI parsing; adjust STREAM / WAIT_SECONDS / MAX_NEW_TOKENS / TEMPERATURE above."""


"""helper only"""


if __name__ == "__main__":
    asyncio.run(main())
