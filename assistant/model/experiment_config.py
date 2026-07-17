from __future__ import annotations

import attrs
from sivkit import Entity, entity_type


@entity_type
class ExperimentConfig(Entity):
    """A loaded experiment configuration.

    Holds the full config data parsed from the experiment JSON file.
    Stored in the config store with id = "exp-{file_stem}".
    """

    id: str = ""
    name: str = ""
    path: str = ""
    data_loader: str = ""
    prompt: str = ""
    dataset_path: str = ""
    prompt_template: str = ""
    generation_params: dict = attrs.Factory(dict)
    resolution: list | None = None
    start_index: int | None = None
    end_index: int | None = None
    stream: bool = True
    chat_template_params: dict = attrs.Factory(dict)
    extraction: str = "response"  # "response" or "tool_call"
    focus_mode: str = "main"  # which focus mode to target for inference
    # Per-agent cap on tool-execution turns: {agent_name: max_turns}.
    max_turns: dict = attrs.Factory(dict)

    @staticmethod
    def id_for_path(path_stem: str) -> str:
        return f"exp-{path_stem}"
