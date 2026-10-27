"""COCO bounding-box average precision with pycocotools, the reference implementation.

Ground truth is the original annotation file, crowd regions included:
pycocotools ignores the detections that fall on them.
"""

import contextlib
import io
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from canvit_nnx.detect.decode import Detections

log = logging.getLogger(__name__)

STATISTICS = (
    "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
    "AR_max1", "AR_max10", "AR_max100", "AR_small", "AR_medium", "AR_large",
)
"""COCOeval.summarize's statistics, in its order."""
SCORE_THRESHOLD = 1e-3
"""Detections scoring at most this are not scored."""


def load_ground_truth(annotation_file: Path) -> COCO:
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(annotation_file))


def coco_detections(
    detections: Detections[np.ndarray],
    *,
    image_ids: np.ndarray,
    heights: np.ndarray,
    widths: np.ndarray,
    category_ids: Sequence[int],
) -> np.ndarray:
    """Host detections of [N] images as pycocotools' [M, 7] rows: image id, x, y, width, height, score, category.

    Boxes leave the normalized frame by each image's own height and width;
    boxes of zero or negative extent are dropped, since they can only miss.
    """
    heights, widths = heights[:, None].astype(np.float64), widths[:, None].astype(np.float64)
    boxes = np.asarray(detections.boxes, dtype=np.float64)
    top, left = boxes[..., 0] * heights, boxes[..., 1] * widths
    height, width = (boxes[..., 2] - boxes[..., 0]) * heights, (boxes[..., 3] - boxes[..., 1]) * widths
    scores = np.asarray(detections.scores, dtype=np.float64)
    keep = (scores > SCORE_THRESHOLD) & (width > 0) & (height > 0)
    image = np.broadcast_to(image_ids[:, None], scores.shape).astype(np.float64)
    category = np.asarray(category_ids, dtype=np.float64)[np.asarray(detections.labels)]
    return np.stack([column[keep] for column in (image, left, top, width, height, scores, category)], axis=1)


def coco_statistics(rows: np.ndarray, *, ground_truth: COCO, image_ids: np.ndarray) -> dict[str, float]:
    """The STATISTICS of [M, 7] detection rows, over the given images only; zero when there are no rows."""
    if not len(rows):
        log.warning("No detections above SCORE_THRESHOLD with a positive extent on %d images", len(image_ids))
        return dict.fromkeys(STATISTICS, 0.0)
    with contextlib.redirect_stdout(io.StringIO()):
        evaluation = COCOeval(ground_truth, ground_truth.loadRes(rows), "bbox")  # pyright: ignore[reportArgumentType]
        evaluation.params.imgIds = [int(image_id) for image_id in image_ids]
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()
    return dict(zip(STATISTICS, map(float, evaluation.stats), strict=True))


def bootstrap_interval(values: Sequence[float], *, resamples: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """The 95% percentile bootstrap interval of the mean of independent values; NaNs for a single value."""
    if len(values) == 1:
        return float("nan"), float("nan")
    sample = np.asarray(values, dtype=np.float64)
    means = np.random.default_rng(seed).choice(sample, size=(resamples, len(sample)), replace=True).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)
