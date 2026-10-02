"""Experiment result persistence helpers — image saving and JSON summaries."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

from assistant.experiments.evaluation import IOU_CORRECT_THRESHOLD

log = logging.getLogger(__name__)


def save_sample_image(output_dir: Path, image: Image.Image, result: dict) -> None:
    """Save the sample image with GT and prediction bboxes overlaid."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size

    # Draw ground truth in green
    gt_bbox = result.get("gt_bbox")
    if gt_bbox:
        x1 = int(gt_bbox[0] / 1000 * img_w)
        y1 = int(gt_bbox[1] / 1000 * img_h)
        x2 = int(gt_bbox[2] / 1000 * img_w)
        y2 = int(gt_bbox[3] / 1000 * img_h)
        if x1 <= x2 and y1 <= y2:
            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)

    # Draw prediction in red (incorrect) or blue (correct)
    pred_bbox = result.get("predicted_bbox")
    if pred_bbox:
        x1 = int(pred_bbox[0] / 1000 * img_w)
        y1 = int(pred_bbox[1] / 1000 * img_h)
        x2 = int(pred_bbox[2] / 1000 * img_w)
        y2 = int(pred_bbox[3] / 1000 * img_h)
        color = "blue" if result.get("correct") else "red"
        if x1 <= x2 and y1 <= y2:
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    else:
        # No bbox could be parsed from the response — make that explicit
        # so a "no shape" result is distinguishable from a wrongly-placed one.
        draw.text((8, 8), "NO SHAPE DETECTED", fill="red")

    # Save to appropriate subfolder
    subdir = "correct" if result.get("correct") else "incorrect"
    filename = f"step_{result['step']:04d}_iou{result['iou']:.2f}.png"
    save_path = output_dir / subdir / filename
    img.save(save_path)


def save_results_json(
    output_dir: Path, config_name: str, agent: str, results: list[dict]
) -> None:
    """Save accumulated results to a JSON summary file."""
    if not results:
        return

    output = {
        "experiment": config_name,
        "agent": agent,
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(results),
        "correct_count": sum(1 for r in results if r.get("correct")),
        "center_in_gt_count": sum(1 for r in results if r.get("center_in_gt")),
        "mean_iou": round(sum(r["iou"] for r in results) / len(results), 4),
        "iou_threshold": IOU_CORRECT_THRESHOLD,
        "results": results,
    }
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(output, indent=2))
    log.info(f"Results saved to {results_path}")


def append_result_jsonl(output_dir: Path, result: dict) -> None:
    """Append a single result to results.jsonl (incremental saving)."""
    jsonl_path = output_dir / "results.jsonl"
    with jsonl_path.open("a") as f:
        f.write(json.dumps(result) + "\n")


def save_run_params(output_dir: Path, params: dict) -> None:
    """Persist the model call parameters for a run to run_params.json."""
    (output_dir / "run_params.json").write_text(json.dumps(params, indent=2))
