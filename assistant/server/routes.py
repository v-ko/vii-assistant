from typing import Any

import sivkit
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

    sivkit.call_delayed(terminal_actions.toggle_terminal, 0)
    return CommandResponse(success=True, message="Terminal toggled")


@router.post("/stop", response_model=CommandResponse)
def stop_assistant() -> CommandResponse:
    """Interrupt assistant execution."""
    from assistant.actions import stop_assistant as _stop

    sivkit.call_delayed(_stop, 0)
    return CommandResponse(success=True, message="Assistant stopped")


@router.post("/toggle_recording", response_model=CommandResponse)
def toggle_recording() -> CommandResponse:
    """Toggle audio recording for transcription."""
    from assistant.recording_procedures import toggle_recording as _toggle

    sivkit.call_delayed(_toggle, 0)
    return CommandResponse(success=True, message="Recording toggled")


@router.post("/snippet", response_model=CommandResponse)
def take_snippet() -> CommandResponse:
    """Activate screen snippet overlay for region selection."""
    from assistant.snippet_procedures import start_snippet

    sivkit.call_delayed(start_snippet, 0)
    return CommandResponse(success=True, message="Snippet overlay activated")


# ── Supervised mode routes ───────────────────────────────────────


@router.post("/supervised/correct", response_model=CommandResponse)
def supervised_correct() -> CommandResponse:
    """Supervisor marks current turn as correct."""
    gate = vii.project_manager.hybrid_segment_service.supervised_gate
    sivkit.call_delayed(gate.submit_correct, 0)
    return CommandResponse(success=True, message="Marked correct")


@router.post("/supervised/pass", response_model=CommandResponse)
def supervised_pass() -> CommandResponse:
    """Supervisor passes (no label)."""
    gate = vii.project_manager.hybrid_segment_service.supervised_gate
    sivkit.call_delayed(gate.submit_pass, 0)
    return CommandResponse(success=True, message="Marked pass")


class SupervisedErrorRequest(BaseModel):
    correction: str


@router.post("/supervised/error", response_model=CommandResponse)
def supervised_error(body: SupervisedErrorRequest) -> CommandResponse:
    """Supervisor marks turn as error with correction text."""
    gate = vii.project_manager.hybrid_segment_service.supervised_gate
    sivkit.call_delayed(gate.submit_error, 0, args=[body.correction])
    return CommandResponse(success=True, message="Marked error")
