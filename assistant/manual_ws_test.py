"""Manual websocket test: send prompt + image + request and print responses.

Usage:
  python -m assistant.manual_ws_test --url ws://desk_local:8000/ws/context
"""

from __future__ import annotations

import asyncio
import base64
import json
from io import BytesIO

import numpy as np
import websockets
from fusion.storage.ws_sync_service import WebSocketSyncService
from PIL import Image

from assistant.inference.context import ContextItem  # needed to init registry
from assistant.inference.context_store import ContextStore

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
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


async def main() -> None:
    store = ContextStore()

    # Print any changes received from the server
    def on_changes(delta, origin=None):
        for key, change_data in delta.asdict().items():
            eid, reverse, forward = change_data
            print(f"<- {key}: {json.dumps(forward)[:500]}")

    store.on_changes = on_changes

    sync = WebSocketSyncService(store, role="receiver")

    async with websockets.connect("ws://localhost:8000/ws/context") as ws:
        print("Connected. Running sync as receiver...")

        async def send(msg: dict) -> None:
            await ws.send(json.dumps(msg))

        async def receive() -> dict:
            raw = await ws.recv()
            return json.loads(raw)

        # Start sync in a background task
        sync_task = asyncio.create_task(sync.run(send, receive))

        # Wait a moment for full_state exchange
        await asyncio.sleep(1.0)

        # 1) Prompt item
        prompt_item = ContextItem()
        prompt_item.position = 0
        prompt_item.content = {"text": "Describe the dominant colors in this image."}
        store.insert_one(prompt_item)

        # 2) Image item
        image_b64 = build_gradient_b64(256)
        image_item = ContextItem()
        image_item.position = 1
        image_item.content = {"image": image_b64}
        store.insert_one(image_item)

        # 3) Request item
        request_item = ContextItem()
        request_item.position = 2
        request_item.content = {"text": "Please answer now."}
        request_item.request = {
            "stream": STREAM,
            "generation_params": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": TEMPERATURE,
            },
        }
        store.insert_one(request_item)

        print("Sent prompt, image and request. Waiting for responses...")
        await asyncio.sleep(WAIT_SECONDS)
        sync_task.cancel()
        try:
            await ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
