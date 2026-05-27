from typing import Any

import fusion
from fastapi import APIRouter
from pydantic import BaseModel

import assistant.terminal_actions as terminal_actions
from assistant.facade import vii

# Create the FastAPI app
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

    fusion.call_delayed(terminal_actions.toggle_terminal, 0)
    return CommandResponse(success=True, message="Terminal toggled")


@router.post("/confirm", response_model=CommandResponse)
def confirm_action() -> CommandResponse:
    """Confirm pending assistant actions (USER_APPROVE mode)."""
    gate = vii.project_manager.hybrid_segment_service.action_gate
    fusion.call_delayed(gate.confirm, 0)
    return CommandResponse(success=True, message="Action confirmed")


@router.post("/stop", response_model=CommandResponse)
def stop_assistant() -> CommandResponse:
    """Interrupt assistant execution."""
    from assistant.actions import stop_assistant as _stop

    fusion.call_delayed(_stop, 0)
    return CommandResponse(success=True, message="Assistant stopped")


@router.post("/toggle_recording", response_model=CommandResponse)
def toggle_recording() -> CommandResponse:
    """Toggle audio recording for transcription."""
    from assistant.recording_procedures import toggle_recording as _toggle

    fusion.call_delayed(_toggle, 0)
    return CommandResponse(success=True, message="Recording toggled")


@router.post("/snippet", response_model=CommandResponse)
def take_snippet() -> CommandResponse:
    """Activate screen snippet overlay for region selection."""
    from assistant.snippet_procedures import start_snippet

    fusion.call_delayed(start_snippet, 0)
    return CommandResponse(success=True, message="Snippet overlay activated")
