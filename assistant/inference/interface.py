from __future__ import annotations

from typing import Any, Literal, TypedDict

from fusion.libs.entity.change import Change

# Message type literals
MsgType = Literal["Change", "AppendItemContentText"]


class AppendItemContentTextPayload(TypedDict):
    item_id: str
    text: str


def wrap_change(change: Change) -> dict[str, Any]:
    return {"type": "Change", "payload": change.as_safe_delta_dict()}


def wrap_append_item_content_text(item_id: str, text: str) -> dict[str, Any]:
    payload: AppendItemContentTextPayload = {"item_id": item_id, "text": text}
    return {"type": "AppendItemContentText", "payload": payload}


def unwrap_message(raw: dict[str, Any]) -> tuple[str, Any]:
    """Return (type, payload). Minimal validation only."""
    mtype = raw.get("type")
    if not isinstance(mtype, str):
        raise ValueError("Missing message type")
    return mtype, raw.get("payload")
