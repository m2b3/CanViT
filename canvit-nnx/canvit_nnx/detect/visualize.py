"""Pictures of a detector's rollout: after each glimpse, its viewpoint and the boxes it predicts, beside the truth."""

import numpy as np
from PIL import Image, ImageDraw

from canvit_nnx.detect.evaluate import Rollout

TRUTH_COLOR = (0, 255, 0)
PREDICTION_COLOR = (255, 0, 0)
VIEWPOINT_COLOR = (255, 255, 0)
MIN_SCORE = 0.3
"""Predictions scoring below this are not drawn."""


def _box(box: np.ndarray, size: int) -> list[float]:
    """Normalized (ymin, xmin, ymax, xmax) to PIL's [x0, y0, x1, y1], ordered even when the prediction is inverted."""
    x0, x1 = sorted((float(box[1]) * size, float(box[3]) * size))
    y0, y1 = sorted((float(box[0]) * size, float(box[2]) * size))
    return [x0, y0, x1, y1]


def rollout_panels(rollout: Rollout, *, category_names: tuple[str, ...]) -> Image.Image:
    """The rollout's scene once per glimpse, left to right, outlining the glimpse's viewpoint, the true boxes and
    the boxes predicted after the glimpse."""
    size = rollout.scene.shape[0]
    panels = Image.new("RGB", (size * len(rollout.scales), size))
    for glimpse, (center, scale) in enumerate(zip(rollout.centers, rollout.scales, strict=True)):
        panel = Image.fromarray(rollout.scene)
        draw = ImageDraw.Draw(panel)
        for box, label in zip(rollout.image.boxes, rollout.image.labels, strict=True):
            draw.rectangle(_box(box, size), outline=TRUTH_COLOR, width=2)
            draw.text((float(box[1]) * size + 2, float(box[0]) * size + 1), category_names[label], fill=TRUTH_COLOR)
        detections = rollout.detections
        for box, score, label in zip(
            detections.boxes[glimpse], detections.scores[glimpse], detections.labels[glimpse], strict=True
        ):
            if score < MIN_SCORE:
                break
            x0, _, _, y1 = _box(box, size)
            draw.rectangle(_box(box, size), outline=PREDICTION_COLOR, width=2)
            draw.text((x0 + 2, y1 - 11), f"{category_names[label]} {score:.2f}", fill=PREDICTION_COLOR)
        viewpoint = np.array([*(center - scale), *(center + scale)])
        draw.rectangle(_box((viewpoint + 1) / 2, size), outline=VIEWPOINT_COLOR, width=3)
        panels.paste(panel, (glimpse * size, 0))
    return panels
