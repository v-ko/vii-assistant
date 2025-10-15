from __future__ import annotations

import time
from pathlib import Path

from assistant.services.session_recorder import SessionRecorder


class _NoopCallback:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self, event: dict):  # type: ignore[override]
        self.count += 1


def measure(label: str, fn):
    t0 = time.perf_counter()
    result = fn()
    dt = (time.perf_counter() - t0) * 1000
    print(f"{label}: {dt:.2f} ms")
    return result, dt


def run(iterations: int = 3, runtime: float = 0.25) -> None:
    print(
        "Running SessionRecorder start/stop benchmark:"
        f" iterations={iterations} runtime={runtime}s"
    )
    cb = _NoopCallback()
    start_times: list[float] = []
    stop_times: list[float] = []

    for i in range(iterations):
        rec = SessionRecorder(event_callback=cb)
        _, dt_start = measure(f"[{i+1}] start", rec.start)
        start_times.append(dt_start)
        # Let it collect events for a bit (user can move mouse / press keys) but keep short
        time.sleep(runtime)
        _, dt_stop = measure(f"[{i+1}] stop", rec.stop)
        stop_times.append(dt_stop)
        # Small gap to avoid overlap
        time.sleep(0.1)

    def stats(label: str, xs: list[float]):
        if not xs:
            return "n/a"
        return f"min={min(xs):.2f}ms avg={sum(xs)/len(xs):.2f}ms max={max(xs):.2f}ms"

    print("--- Summary ---")
    print("start times:", stats("start", start_times))
    print("stop times:", stats("stop", stop_times))
    print(f"events captured: {cb.count}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument(
        "--runtime", type=float, default=0.25, help="seconds to record per iteration"
    )
    args = ap.parse_args()
    run(iterations=args.iterations, runtime=args.runtime)
