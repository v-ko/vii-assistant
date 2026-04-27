from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

import assistant.terminal_actions as terminal_actions
from assistant.registries.actions import execute_on_main_thread

# Create the FastAPI appw
router = APIRouter()


class CommandResponse(BaseModel):
    """Model for command responses."""

    success: bool
    message: str


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    return {"status": "ok", "service": "screenshot-assistant", "version": "0.1.0"}


@router.post("/toggle_terminal", response_model=CommandResponse)
def toggle_terminal() -> CommandResponse:
    """Toggle the terminal window."""

    execute_on_main_thread(terminal_actions.toggle_terminal)
    return CommandResponse(success=True, message="Terminal toggled")
