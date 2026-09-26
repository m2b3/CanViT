"""Pin _batch_confusion against an integer-exact numpy reference.

scatter_add on the GPU is non-deterministic in general, but the integer-valued
accumulator here makes the result associative — bit-identical across runs.
"""

import numpy as np
import pytest
import torch

from canvit_eval.tasks.ade20k_obj.iou import _batch_confusion

try:
    from canvit_specialize.datasets.ade20k import IGNORE_LABEL, NUM_CLASSES
except ImportError:  # pragma: no cover
    IGNORE_LABEL = 255
    NUM_CLASSES = 150


def _numpy_reference(preds: np.ndarray, masks: np.ndarray, n_classes: int):
    """Per-image confusion via np.bincount."""
    B = preds.shape[0]
    inter = np.zeros((B, n_classes), dtype=np.int64)
    union = np.zeros_like(inter)
    gt_area = np.zeros_like(inter)
    for b in range(B):
        valid = masks[b] != IGNORE_LABEL
        p = preds[b][valid].astype(np.int64)
        m = masks[b][valid].astype(np.int64)
        cm = np.bincount(p * n_classes + m, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
        inter[b] = np.diag(cm)
        row = cm.sum(axis=1)
        col = cm.sum(axis=0)
        union[b] = row + col - inter[b]
        gt_area[b] = col
    return inter, union, gt_area


def _assert_same(t: torch.Tensor, np_ref: np.ndarray, label: str) -> None:
    assert tuple(t.shape) == np_ref.shape, (label, t.shape, np_ref.shape)
    assert np.array_equal(t.long().cpu().numpy(), np_ref), label


@pytest.mark.parametrize(
    "shape, n_classes",
    [
        ((4, 32, 32), 10),
        ((4, 128, 128), 50),
        ((2, 512, 512), 150),
    ],
)
def test_batch_matches_numpy_exactly(shape, n_classes):
    torch.manual_seed(0)
    preds = torch.randint(0, n_classes, shape, dtype=torch.int64)
    masks = torch.randint(0, n_classes, shape, dtype=torch.int64)
    masks[torch.rand(shape) < 0.10] = IGNORE_LABEL

    np_i, np_u, np_g = _numpy_reference(preds.numpy(), masks.numpy(), n_classes)
    i, u, g = _batch_confusion(preds, masks, n_classes)
    _assert_same(i, np_i, "inter")
    _assert_same(u, np_u, "union")
    _assert_same(g, np_g, "gt_area")


def test_all_ignore_label():
    """Every pixel ignored → zero counts everywhere."""
    preds = torch.zeros((3, 16, 16), dtype=torch.int64)
    masks = torch.full((3, 16, 16), IGNORE_LABEL, dtype=torch.int64)
    i, u, g = _batch_confusion(preds, masks, 5)
    assert i.sum() == 0 and u.sum() == 0 and g.sum() == 0


def test_perfect_prediction():
    """pred == mask → inter == union == gt_area."""
    n_classes = 7
    masks = torch.randint(0, n_classes, (4, 64, 64), dtype=torch.int64)
    i, u, g = _batch_confusion(masks.clone(), masks, n_classes)
    assert torch.equal(i, g)
    assert torch.equal(u, g)


def test_realistic_ade20k_distribution():
    torch.manual_seed(42)
    shape = (4, 512, 512)
    preds = torch.randint(0, NUM_CLASSES, shape, dtype=torch.int64)
    masks = torch.randint(0, NUM_CLASSES, shape, dtype=torch.int64)
    masks[torch.rand(shape) < 0.30] = IGNORE_LABEL

    np_i, np_u, np_g = _numpy_reference(preds.numpy(), masks.numpy(), NUM_CLASSES)
    i, u, g = _batch_confusion(preds, masks, NUM_CLASSES)
    _assert_same(i, np_i, "inter")
    _assert_same(u, np_u, "union")
    _assert_same(g, np_g, "gt_area")
