"""Hard-coded focus mode registry.

Each focus mode defines:
- system_prompt_path: Path to system prompt file (relative to task folder)
- execution: "server" (inference-based) or "client" (executed locally by HybridSegmentService)
- includes_perception: Whether the latest screenshot is included in context
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Hard-coded task folder path (relative to project root)
TASKS_DIR = Path(__file__).resolve().parent.parent.parent / "tasks"
ACTIVE_TASK = "info_gathering"


@dataclass(frozen=True)
class FocusModeConfig:
    name: str
    execution: str  # "server" or "client"
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
        execution="server",
        includes_perception=True,
    ),
    "localization": FocusModeConfig(
        name="localization",
        execution="server",
        includes_perception=True,
    ),
    "python": FocusModeConfig(
        name="python",
        execution="client",
        includes_perception=False,
        has_prompt_file=False,
    ),
    "click_at": FocusModeConfig(
        name="click_at",
        execution="client",
        includes_perception=False,
        has_prompt_file=False,
    ),
    "scroll": FocusModeConfig(
        name="scroll",
        execution="client",
        includes_perception=False,
        has_prompt_file=False,
    ),
}

# Focus modes executed client-side (not via server inference)
CLIENT_TOOLS: set[str] = {
    name for name, cfg in FOCUS_MODES.items() if cfg.execution == "client"
}

# Max tool-call chain depth before force-stopping
MAX_TOOL_CALL_DEPTH = 20
