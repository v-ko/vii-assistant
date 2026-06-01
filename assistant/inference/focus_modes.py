"""Focus mode registry and client-tool declarations.

Focus modes define server-side inference contexts (system prompts, perception).
Client tools are executed locally by HybridSegmentService.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Hard-coded task folder path (relative to project root)
TASKS_DIR = Path(__file__).resolve().parent.parent.parent / "tasks"
ACTIVE_TASK = "info_gathering"


@dataclass(frozen=True)
class FocusModeConfig:
    name: str
    includes_perception: bool = False
    has_prompt_file: bool = True  # Whether this mode has a system prompt file

    def system_prompt_path(self, task: str = ACTIVE_TASK) -> Path:
        return TASKS_DIR / task / f"{self.name}.md"

    def load_system_prompt(self, task: str = ACTIVE_TASK) -> str:
        if not self.has_prompt_file:
            return ""
        path = self.system_prompt_path(task)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8").strip()


FOCUS_MODES: dict[str, FocusModeConfig] = {
    "main": FocusModeConfig(
        name="main",
        includes_perception=True,
    ),
    "localization": FocusModeConfig(
        name="localization",
        includes_perception=True,
    ),
}

# Tools executed client-side (not focus modes — just tool calls routed to the client)
CLIENT_TOOLS: set[str] = {"python", "click_at", "scroll"}


# Max tool-call chain depth before force-stopping
MAX_TOOL_CALL_DEPTH = 20
