from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any

import assistant.actions as actions

# Create the FastAPI appw
router = APIRouter()


class CommandResponse(BaseModel):
    """Model for command responses."""
    success: bool
    message: str


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "screenshot-assistant",
        "version": "0.1.0"
    }


@router.post("/toggle_terminal", response_model=CommandResponse)
def toggle_terminal() -> CommandResponse:
    """Toggle the terminal window."""

    actions.execute_on_main_thread(actions.toggle_terminal)
    return CommandResponse(success=True, message="Terminal toggled")
