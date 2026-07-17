"""Focus mode registry with agent-based prompt loading.

Each focus mode defines:
- system_prompt_path: Path to system prompt file (resolved from active agent folder)
- execution: "server" (inference-based) or "client" (executed locally by HybridSegmentService)
- includes_perception: Whether the latest screenshot is included in context
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from assistant.constants import AGENTS_DIR


@dataclass(frozen=True)
class FocusModeConfig:
    name: str
    execution: str  # "server" or "client"
    includes_perception: bool = False
    has_prompt_file: bool = True  # Whether this mode has a system prompt file

    def system_prompt_path(self, agent: str) -> Path:
        """Resolve prompt path from agent folder. Raises if not found."""
        agent_path = AGENTS_DIR / agent / f"{self.name}.md"
        if not agent_path.exists():
            raise FileNotFoundError(f"System prompt not found: {agent_path}")
        return agent_path

    def load_system_prompt(self, agent: str) -> str:
        if not self.has_prompt_file:
            return ""
        path = self.system_prompt_path(agent)
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

# Focus modes whose context should include the latest screenshot
PERCEPTION_MODES: list[str] = [
    name for name, cfg in FOCUS_MODES.items() if cfg.includes_perception
]

# Max tool-call chain depth before force-stopping
MAX_TOOL_CALL_DEPTH = 5
