"""Input control via xdotool (scroll) + ydotool (mouse move, type, click).

Uses relative mouse movement with iterative convergence (via Qt cursor
position feedback) to work correctly across multiple monitors.
Requires ydotoold daemon running for mouse/type. All operations are async.
"""

from __future__ import annotations

import asyncio
import shutil

from fusion.logging import get_logger
from PySide6.QtGui import QCursor

log = get_logger(__name__)

_MAX_SCROLL_STEPS = 20
_MOVE_MAX_ATTEMPTS = 8
_MOVE_TOLERANCE = 2  # pixels


def _check_available(cmd: str) -> bool:
    """Check that a binary exists on PATH."""
    if shutil.which(cmd) is None:
        log.error("%s not found on PATH.", cmd)
        return False
    return True


async def _run(cmd: str, args: list[str]) -> bool:
    """Run a subprocess command. Returns True on success."""
    log.info("%s %s", cmd, " ".join(args))
    proc = await asyncio.create_subprocess_exec(
        cmd,
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    out = stdout.decode().strip() if stdout else ""
    err = stderr.decode().strip() if stderr else ""
    if out or err or proc.returncode != 0:
        log.info("%s rc=%d stdout=%r stderr=%r", cmd, proc.returncode, out, err)
    if proc.returncode != 0:
        return False
    return True


async def move_pointer(x: int, y: int) -> bool:
    """Move pointer to absolute virtual-desktop coordinates.

    Uses iterative relative moves with Qt cursor feedback to handle
    mouse acceleration and work across multiple monitors.
    """
    if not _check_available("xdotool"):
        return False

    for i in range(_MOVE_MAX_ATTEMPTS):
        pos = QCursor.pos()
        dx = x - pos.x()
        dy = y - pos.y()
        if abs(dx) <= _MOVE_TOLERANCE and abs(dy) <= _MOVE_TOLERANCE:
            return True
        if not await _run("xdotool", ["mousemove_relative", "--", str(dx), str(dy)]):
            return False
        await asyncio.sleep(0.02)

    pos = QCursor.pos()
    if abs(pos.x() - x) <= _MOVE_TOLERANCE and abs(pos.y() - y) <= _MOVE_TOLERANCE:
        return True
    log.warning(
        "move_pointer: failed to converge after %d attempts "
        "(target=%d,%d actual=%d,%d)",
        _MOVE_MAX_ATTEMPTS,
        x,
        y,
        pos.x(),
        pos.y(),
    )
    return False


async def click(x: int, y: int, button: int = 0) -> bool:
    """Move pointer to (x, y) and click. button: 0=left, 1=right, 2=middle."""
    if not await move_pointer(x, y):
        return False
    # xdotool: 1=left, 2=middle, 3=right
    xdo_button = {0: 1, 1: 3, 2: 2}.get(button, 1)
    return await _run("xdotool", ["click", str(xdo_button)])


async def scroll(steps: int) -> bool:
    """Scroll at current pointer position. Positive=up, negative=down."""
    if not _check_available("xdotool"):
        return False
    if steps == 0:
        return True
    if abs(steps) > _MAX_SCROLL_STEPS:
        log.warning("scroll: clamped %d to %d", steps, _MAX_SCROLL_STEPS)
        steps = _MAX_SCROLL_STEPS if steps > 0 else -_MAX_SCROLL_STEPS
    # xdotool: button 4=up, 5=down
    button = "4" if steps > 0 else "5"
    count = abs(steps)
    return await _run("xdotool", ["click", "--repeat", str(count), button])


async def type_text(text: str) -> bool:
    """Type text at current focus."""
    if not _check_available("xdotool"):
        return False
    return await _run("xdotool", ["type", "--", text])
