from __future__ import annotations

import attrs
from fusion import Entity, entity_type


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
    generation_params: dict = attrs.Factory(dict)
    resolution: list | None = None
    stream: bool = True
    chat_template_params: dict = attrs.Factory(dict)

    @staticmethod
    def id_for_path(path_stem: str) -> str:
        return f"exp-{path_stem}"
