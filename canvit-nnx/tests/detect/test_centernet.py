import jax
import jax.numpy as jnp
import numpy as np
import pytest

from canvit_nnx.detect.decode import decode
from canvit_nnx.detect.loss import detection_loss, focal_loss
from canvit_nnx.detect.model import CenterNetMaps
from canvit_nnx.detect.targets import Targets, gaussian_radius, render

GRID_SIZE, NUM_CLASSES = 16, 5
# Boxes whose centers fall in distinct cells: (ymin, xmin, ymax, xmax), normalized.
BOXES = np.array([[0.1, 0.2, 0.5, 0.45], [0.55, 0.6, 0.95, 0.9], [0.0, 0.7, 0.2, 1.0]], dtype=np.float32)
LABELS = np.array([3, 0, 3], dtype=np.int32)


def _batched(targets: Targets[np.ndarray]) -> Targets[jax.Array]:
    return Targets(*(jnp.asarray(field)[None] for field in targets))


def test_gaussian_radius_is_centernets() -> None:
    # CenterNet's roots give 0.2733 s for a square of side s cells; the corrected quadratic would give 0.0817 s.
    assert gaussian_radius(100.0, 100.0) == pytest.approx(27.33, abs=0.01)
    assert gaussian_radius(1.0, 1.0) == 1.0


def test_targets_peak_at_each_center_with_its_size_and_offset() -> None:
    targets = render(BOXES, LABELS, grid_size=GRID_SIZE, num_classes=NUM_CLASSES)
    rows, cols = np.nonzero(targets.mask)
    assert len(rows) == len(BOXES)
    for box, label in zip(BOXES * GRID_SIZE, LABELS, strict=True):
        center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        row, col = int(center[0]), int(center[1])
        assert targets.mask[row, col] == 1.0 and targets.heatmap[row, col, label] == 1.0
        np.testing.assert_allclose(targets.size[row, col], (box[2] - box[0], box[3] - box[1]), rtol=1e-6)
        np.testing.assert_allclose(targets.offset[row, col], (center[0] - row, center[1] - col), atol=1e-6)
    assert np.sum(targets.heatmap == 1.0) == len(BOXES), "a peak of 1 at each center and nowhere else"


def test_decoding_targets_recovers_the_boxes() -> None:
    targets = render(BOXES, LABELS, grid_size=GRID_SIZE, num_classes=NUM_CLASSES)
    probabilities = np.clip(targets.heatmap, 1e-6, 1 - 1e-6)
    maps = CenterNetMaps(
        heatmap=jnp.log(probabilities / (1 - probabilities))[None],
        size=jnp.asarray(targets.size)[None],
        offset=jnp.asarray(targets.offset)[None],
    )
    detections = decode(maps)
    found = int(np.sum(np.asarray(detections.scores[0]) > 1 - 1e-5))
    assert found == len(BOXES)
    order = np.argsort(BOXES[:, 0])
    recovered = np.asarray(detections.boxes[0, :found])
    np.testing.assert_allclose(recovered[np.argsort(recovered[:, 0])], BOXES[order], atol=1e-6)
    np.testing.assert_array_equal(np.sort(np.asarray(detections.labels[0, :found])), np.sort(LABELS))


def test_the_loss_vanishes_at_the_targets_and_stays_finite_without_objects() -> None:
    targets = _batched(render(BOXES, LABELS, grid_size=GRID_SIZE, num_classes=NUM_CLASSES))
    confident = jnp.where(targets.heatmap == 1.0, 30.0, -30.0)
    loss, terms = detection_loss(
        CenterNetMaps(heatmap=confident, size=targets.size, offset=targets.offset), targets, size_weight=0.4
    )
    assert float(loss) < 1e-6 and float(terms.size) == 0.0 and float(terms.offset) == 0.0

    empty = _batched(render(BOXES[:0], LABELS[:0], grid_size=GRID_SIZE, num_classes=NUM_CLASSES))
    gradient = jax.grad(focal_loss)(jnp.zeros_like(empty.heatmap), empty.heatmap)
    assert np.all(np.isfinite(np.asarray(gradient)))
