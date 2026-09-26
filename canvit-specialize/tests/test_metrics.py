"""Tests for mIoUAccumulator — pure CPU, no dataset needed."""

import torch
from canvit_specialize.metrics import mIoUAccumulator


def _reference_update(num_classes: int, ignore_index: int, preds: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Naive per-image, per-class reference for the bincount-based update.

    Test-only; pins bit-equivalence against the production vectorized path.
    """
    intersection = torch.zeros(num_classes)
    union = torch.zeros(num_classes)
    for i in range(preds.shape[0]):
        mask = targets[i] != ignore_index
        p, t = preds[i][mask], targets[i][mask]
        for cls in range(num_classes):
            p_cls = p == cls
            t_cls = t == cls
            intersection[cls] += (p_cls & t_cls).sum()
            union[cls] += (p_cls | t_cls).sum()
    return intersection, union


def test_perfect_predictions() -> None:
    acc = mIoUAccumulator(num_classes=3, ignore_index=255, device=torch.device("cpu"))
    targets = torch.tensor([[[0, 1, 2, 0]]])  # [1, 1, 4]
    acc.update(targets, targets)
    assert acc.compute() > 0.99


def test_reset_zeros_counters() -> None:
    acc = mIoUAccumulator(num_classes=3, ignore_index=255, device=torch.device("cpu"))
    t = torch.tensor([[[0, 1, 2]]])  # [1, 1, 3]
    acc.update(t, t)
    assert acc.compute() > 0.99
    acc.reset()
    assert acc.intersection.sum().item() == 0
    assert acc.union.sum().item() == 0


def test_ignore_index() -> None:
    acc = mIoUAccumulator(num_classes=3, ignore_index=255, device=torch.device("cpu"))
    preds = torch.tensor([[[0, 1, 2, 0]]])  # [1, 1, 4]
    targets = torch.tensor([[[0, 1, 255, 0]]])
    acc.update(preds, targets)
    miou = acc.compute()
    assert miou > 0.5


def test_vectorized_matches_reference_random_data() -> None:
    """Bincount-based update must produce bit-identical counters to the naive reference.

    Tests at ADE20K-realistic shapes (B=4, 64x64 spatial, 150 classes, ~10% ignore_index)
    to exercise the actual workload, not a toy case.
    """
    torch.manual_seed(0)
    num_classes = 150
    ignore_index = 255
    B, H, W = 4, 64, 64

    preds = torch.randint(0, num_classes, (B, H, W))
    targets = torch.randint(0, num_classes, (B, H, W))
    # Inject ~10% ignore_index into targets
    ignore_mask = torch.rand(B, H, W) < 0.1
    targets[ignore_mask] = ignore_index

    ref_int, ref_uni = _reference_update(num_classes, ignore_index, preds, targets)

    acc = mIoUAccumulator(num_classes, ignore_index, device=torch.device("cpu"))
    acc.update(preds, targets)

    assert torch.equal(acc.intersection, ref_int), \
        f"intersection mismatch: max abs diff = {(acc.intersection - ref_int).abs().max().item()}"
    assert torch.equal(acc.union, ref_uni), \
        f"union mismatch: max abs diff = {(acc.union - ref_uni).abs().max().item()}"


def test_vectorized_matches_reference_all_ignore() -> None:
    """Edge case: all targets are ignore_index → both counters must stay zero."""
    num_classes = 5
    ignore_index = 255
    preds = torch.randint(0, num_classes, (2, 8, 8))
    targets = torch.full((2, 8, 8), ignore_index, dtype=torch.int64)

    acc = mIoUAccumulator(num_classes, ignore_index, device=torch.device("cpu"))
    acc.update(preds, targets)
    assert acc.intersection.sum().item() == 0
    assert acc.union.sum().item() == 0


def test_vectorized_matches_reference_perfect_predictions_with_ignore() -> None:
    """Perfect predictions on the non-ignored region → intersection == union per present class."""
    num_classes = 4
    ignore_index = 255
    targets = torch.tensor([[[0, 1, 2, 3, 0, 1, ignore_index, ignore_index]]])  # [1, 1, 8]
    preds = torch.tensor([[[0, 1, 2, 3, 0, 1, 0, 0]]])  # match valid region; arbitrary on ignored

    ref_int, ref_uni = _reference_update(num_classes, ignore_index, preds, targets)
    acc = mIoUAccumulator(num_classes, ignore_index, device=torch.device("cpu"))
    acc.update(preds, targets)

    assert torch.equal(acc.intersection, ref_int)
    assert torch.equal(acc.union, ref_uni)


def test_accumulation_across_multiple_updates() -> None:
    """Two consecutive update() calls must give the same counters as one big update on concatenated data."""
    torch.manual_seed(1)
    num_classes = 10
    ignore_index = 255
    B, H, W = 3, 16, 16

    preds_a = torch.randint(0, num_classes, (B, H, W))
    targets_a = torch.randint(0, num_classes, (B, H, W))
    preds_b = torch.randint(0, num_classes, (B, H, W))
    targets_b = torch.randint(0, num_classes, (B, H, W))

    acc_split = mIoUAccumulator(num_classes, ignore_index, device=torch.device("cpu"))
    acc_split.update(preds_a, targets_a)
    acc_split.update(preds_b, targets_b)

    acc_joint = mIoUAccumulator(num_classes, ignore_index, device=torch.device("cpu"))
    acc_joint.update(torch.cat([preds_a, preds_b]), torch.cat([targets_a, targets_b]))

    assert torch.equal(acc_split.intersection, acc_joint.intersection)
    assert torch.equal(acc_split.union, acc_joint.union)
