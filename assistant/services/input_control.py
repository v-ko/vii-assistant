"""Input control via ydotool — moves pointer, clicks, scrolls, types.

Requires ydotoold daemon running. All operations are async (non-blocking).
"""

from __future__ import annotations

import asyncio
import shutil

from fusion.logging import get_logger

log = get_logger(__name__)

_MAX_SCROLL_STEPS = 20


def _check_ydotool_available() -> bool:
    """Check that ydotool binary exists on PATH."""
    if shutil.which("ydotool") is None:
        log.error(
            "ydotool not found on PATH. "
            "Install it (e.g. `sudo pacman -S ydotool`) and ensure "
            "`ydotoold` is running (`systemctl --user start ydotool`)."
        )
        return False
    return True


async def _run(args: list[str]) -> bool:
    """Run a ydotool command. Returns True on success."""
    proc = await asyncio.create_subprocess_exec(
        "ydotool",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        err_msg = stderr.decode().strip() if stderr else "unknown error"
        log.error("ydotool %s failed (rc=%d): %s", args, proc.returncode, err_msg)
        return False
    return True


async def move_pointer(x: int, y: int) -> bool:
    """Move pointer to absolute screen coordinates."""
    if not _check_ydotool_available():
        return False
    return await _run(["mousemove", "--absolute", "-x", str(x), "-y", str(y)])


async def click(x: int, y: int, button: int = 0) -> bool:
    """Move pointer to (x, y) and click. button: 0=left, 1=right, 2=middle."""
    if not _check_ydotool_available():
        return False
    # Move first, then click
    if not await _run(["mousemove", "--absolute", "-x", str(x), "-y", str(y)]):
        return False
    return await _run(["click", hex(button)])


async def scroll(steps: int) -> bool:
    """Scroll at current pointer position. Positive=up, negative=down."""
    if not _check_ydotool_available():
        return False
    if steps == 0:
        return True
    if abs(steps) > _MAX_SCROLL_STEPS:
        log.warning("scroll: clamped %d to %d", steps, _MAX_SCROLL_STEPS)
        steps = _MAX_SCROLL_STEPS if steps > 0 else -_MAX_SCROLL_STEPS
    return await _run(["mousemove", "--wheel", "--", "-y", str(steps)])


async def type_text(text: str) -> bool:
    """Type text at current focus."""
    if not _check_ydotool_available():
        return False
    return await _run(["type", "--", text])
