from __future__ import annotations

import attrs
from sivkit import Entity, entity_type


@entity_type
class ViiConfig(Entity):
    id: str = "app-config"
    capture_screen: str = ""
    selected_model: str = ""
    max_new_tokens: int = 4096
    transcription: dict = attrs.Factory(lambda: {"input_device": ""})

    @property
    def input_device(self) -> str:
        return self.transcription.get("input_device", "")

    @input_device.setter
    def input_device(self, value: str) -> None:
        self.transcription = {**self.transcription, "input_device": value}
