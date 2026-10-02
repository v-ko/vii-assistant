"""Compute experiment statistics from results.json or image filenames.

Usage:
    python -m assistant.experiments.stats <results_dir> [<results_dir2> ...]

If results.json exists in the directory, uses that. Otherwise reconstructs
IoU values from image filenames (step_NNNN_iouX.XX.png) in correct/incorrect/.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from statistics import mean, median


def load_results(results_dir: Path) -> list[dict]:
    """Load results from JSONL, JSON, or reconstruct from image filenames."""
    # Prefer JSONL (incremental, always up to date)
    jsonl_path = results_dir / "results.jsonl"
    if jsonl_path.exists():
        results = []
        for line in jsonl_path.read_text().splitlines():
            line = line.strip()
            if line:
                results.append(json.loads(line))
        return results

    # Fall back to results.json (legacy)
    json_path = results_dir / "results.json"
    if json_path.exists():
        data = json.loads(json_path.read_text())
        return data["results"]

    # Reconstruct from image filenames
    results = []
    pattern = re.compile(r"step_(\d+)_iou(\d+\.\d+)\.png")

    for subdir, correct in [("correct", True), ("incorrect", False)]:
        folder = results_dir / subdir
        if not folder.exists():
            continue
        for img in folder.iterdir():
            m = pattern.match(img.name)
            if m:
                results.append(
                    {
                        "step": int(m.group(1)),
                        "iou": float(m.group(2)),
                        "correct": correct,
                    }
                )

    results.sort(key=lambda r: r["step"])
    return results


def compute_stats(results: list[dict]) -> dict:
    """Compute aggregate statistics from a list of per-sample results."""
    if not results:
        return {"error": "No results"}

    ious = [r["iou"] for r in results]
    total = len(results)
    errors = [r for r in results if r.get("error")]

    stats = {
        "total_samples": total,
        "errors": len(errors),
        "evaluated": total - len(errors),
        "mean_iou": round(mean(ious), 4),
        "median_iou": round(median(ious), 4),
        "accuracy@0.3": round(sum(1 for i in ious if i >= 0.3) / total, 4),
        "accuracy@0.5": round(sum(1 for i in ious if i >= 0.5) / total, 4),
        "accuracy@0.7": round(sum(1 for i in ious if i >= 0.7) / total, 4),
    }

    # IoU histogram (10 buckets: 0.0-0.1, 0.1-0.2, ..., 0.9-1.0)
    histogram = [0] * 10
    for iou in ious:
        bucket = min(int(iou * 10), 9)
        histogram[bucket] += 1
    stats["iou_histogram"] = {
        f"{i/10:.1f}-{(i+1)/10:.1f}": count for i, count in enumerate(histogram)
    }

    return stats


def load_run_params(results_dir: Path) -> dict | None:
    """Load the persisted model call parameters for a run, if available."""
    params_path = results_dir / "run_params.json"
    if not params_path.exists():
        return None
    try:
        return json.loads(params_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _format_run_params(params: dict) -> list[str]:
    """Render the model call parameters as report lines."""
    lines = ["  Model call parameters:"]
    model = params.get("model")
    if model:
        lines.append(f"    model: {model}")
    for key in ("focus_mode", "extraction", "stream"):
        if key in params:
            lines.append(f"    {key}: {params[key]}")
    gen = params.get("generation_params") or {}
    if gen:
        gen_str = ", ".join(f"{k}={v}" for k, v in gen.items())
        lines.append(f"    generation_params: {gen_str}")
    chat = params.get("chat_template_params") or {}
    if chat:
        chat_str = ", ".join(f"{k}={v}" for k, v in chat.items())
        lines.append(f"    chat_template_params: {chat_str}")
    return lines


def format_stats(results_dir: Path, stats: dict, run_params: dict | None = None) -> str:
    """Format statistics as a readable report."""
    lines = [f"=== {results_dir.name} ==="]
    if run_params:
        lines.extend(_format_run_params(run_params))
    lines.append(
        f"  Samples: {stats['total_samples']}  " f"(errors: {stats['errors']})"
    )
    lines.append(f"  Mean IoU:   {stats['mean_iou']:.4f}")
    lines.append(f"  Median IoU: {stats['median_iou']:.4f}")
    lines.append(
        f"  Acc@0.3: {stats['accuracy@0.3']:.2%}  "
        f"Acc@0.5: {stats['accuracy@0.5']:.2%}  "
        f"Acc@0.7: {stats['accuracy@0.7']:.2%}"
    )
    lines.append("  IoU distribution:")
    hist = stats["iou_histogram"]
    max_count = max(hist.values()) if hist else 1
    for bucket, count in hist.items():
        bar = "█" * int(count / max_count * 30) if max_count > 0 else ""
        lines.append(f"    {bucket}: {count:4d} {bar}")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        # Default: show all results directories
        default_dir = Path(__file__).parent / "results"
        if default_dir.exists():
            dirs = sorted(d for d in default_dir.iterdir() if d.is_dir())
        else:
            print("Usage: python -m assistant.experiments.stats <results_dir> [...]")
            sys.exit(1)
    else:
        dirs = [Path(a) for a in sys.argv[1:]]

    for results_dir in dirs:
        if not results_dir.exists():
            print(f"Directory not found: {results_dir}")
            continue
        results = load_results(results_dir)
        if not results:
            print(f"No results found in {results_dir}")
            continue
        stats = compute_stats(results)
        run_params = load_run_params(results_dir)
        print(format_stats(results_dir, stats, run_params))
        print()


if __name__ == "__main__":
    main()
