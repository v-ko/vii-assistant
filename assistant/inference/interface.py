from __future__ import annotations

from typing import Any, Literal, TypedDict, Union

from fusion.libs.entity.change import Change

# Message type literals
MsgType = Literal["Change", "AppendItemContentText"]


class AppendItemContentTextPayload(TypedDict):
    item_id: str
    text: str


class AppendItemContentTextMessage(TypedDict):
    type: Literal["AppendItemContentText"]
    payload: AppendItemContentTextPayload


class ChangeMessage(TypedDict):
    type: Literal["Change"]
    payload: dict[str, Any]


InboundMessage = Union[ChangeMessage, AppendItemContentTextMessage]


def wrap_change(change: Change) -> ChangeMessage:
    change_msg: ChangeMessage = {
        "type": "Change",
        "payload": change.as_safe_delta_dict(),
    }
    return change_msg


def wrap_append_item_content_text(
    item_id: str, text: str
) -> AppendItemContentTextMessage:
    payload: AppendItemContentTextPayload = {"item_id": item_id, "text": text}
    msg: AppendItemContentTextMessage = {
        "type": "AppendItemContentText",
        "payload": payload,
    }
    return msg


def parse_message(raw: dict[str, Any]) -> InboundMessage:
    t = raw.get("type")
    if t == "Change":
        payload = raw.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("Change payload must be dict")
        change_msg: ChangeMessage = {"type": "Change", "payload": payload}
        return change_msg
    if t == "AppendItemContentText":
        payload = raw.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("AppendItemContentText payload must be dict")
        item_id = payload.get("item_id")
        text = payload.get("text")
        if not isinstance(item_id, str) or not isinstance(text, str):
            raise ValueError("AppendItemContentText payload fields invalid")
        frag_msg: AppendItemContentTextMessage = {
            "type": "AppendItemContentText",
            "payload": {"item_id": item_id, "text": text},
        }
        return frag_msg
    raise ValueError(f"Unsupported message type: {t}")
