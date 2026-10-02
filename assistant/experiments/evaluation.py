"""Evaluation utilities for experiment bbox scoring."""

from __future__ import annotations

from sivkit.util.rectangle import Rectangle

from assistant.services.segment_parsing import parse_segment_output

IOU_CORRECT_THRESHOLD = 0.5  # PASCAL VOC standard


def compute_iou(
    box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]
) -> float:
    """Compute IoU between two xyxy boxes (0-1000 grid)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def center_in_box(
    pred_box: tuple[int, int, int, int], gt_box: tuple[int, int, int, int]
) -> bool:
    """Whether the center of pred_box falls inside gt_box (both xyxy)."""
    cx = (pred_box[0] + pred_box[2]) / 2
    cy = (pred_box[1] + pred_box[3]) / 2
    return gt_box[0] <= cx <= gt_box[2] and gt_box[1] <= cy <= gt_box[3]


def parse_bbox_from_response(text: str) -> tuple[int, int, int, int] | None:
    """Extract the first xyxy bbox from a model response.

    Delegates to segment_parsing.parse_segment_output which handles:
    - ```json fenced blocks
    - Bare JSON objects/arrays
    - {"bbox": [...]}, {"bbox_2d": [...]}, {"box_2d": [...]} (Gemma yxyx)
    - bbox(...) / locate(...) function-call syntax
    """
    out = parse_segment_output(text, bbox_format="xyxy")
    if out["bbox"]:
        r: Rectangle = out["bbox"][0]
        # Rectangle stores (x, y, w, h); convert to xyxy
        return (r.x(), r.y(), r.right(), r.bottom())
    return None
