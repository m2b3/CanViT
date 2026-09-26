"""ADE20K metrics against independent references: per-image confusion against numpy's bincount,
and dataset-level MeanIoU against the per-image counts summed over images."""

import numpy as np
import pytest
import torch

from canvit_pytorch.benchmarks.ade20k import IGNORE_LABEL, NUM_CLASSES, MeanIoU, per_image_confusion


def random_labels(
    shape: tuple[int, ...], num_classes: int, *, ignored_fraction: float, seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(predictions, labels), with ignored_fraction of the labels ignored."""
    generator = torch.Generator().manual_seed(seed)
    predictions = torch.randint(0, num_classes, shape, generator=generator)
    labels = torch.randint(0, num_classes, shape, generator=generator)
    labels[torch.rand(shape, generator=generator) < ignored_fraction] = IGNORE_LABEL
    return predictions, labels


def bincount_reference(predictions: np.ndarray, labels: np.ndarray, num_classes: int) -> tuple[np.ndarray, ...]:
    intersection, union, label_area = (np.zeros((len(predictions), num_classes), dtype=np.int64) for _ in range(3))
    for i, (predicted, true) in enumerate(zip(predictions, labels, strict=True)):
        keep = true != IGNORE_LABEL
        confusion = np.bincount(predicted[keep] * num_classes + true[keep], minlength=num_classes**2)
        confusion = confusion.reshape(num_classes, num_classes)  # [predicted, true]
        intersection[i] = np.diag(confusion)
        label_area[i] = confusion.sum(axis=0)
        union[i] = confusion.sum(axis=1) + label_area[i] - intersection[i]
    return intersection, union, label_area


@pytest.mark.parametrize(("shape", "num_classes"), [((4, 32, 32), 10), ((2, 512, 512), NUM_CLASSES)])
def test_per_image_confusion_matches_bincount(shape: tuple[int, ...], num_classes: int) -> None:
    predictions, labels = random_labels(shape, num_classes, ignored_fraction=0.3, seed=0)
    expected = bincount_reference(predictions.numpy(), labels.numpy(), num_classes)
    for name, counts, reference in zip(("intersection", "union", "label area"),
                                       per_image_confusion(predictions, labels, num_classes), expected, strict=True):
        assert np.array_equal(counts.long().numpy(), reference), name


def test_mean_iou_sums_per_image_counts_over_updates() -> None:
    miou = MeanIoU(device=torch.device("cpu"))
    batches = [random_labels((3, 64, 64), NUM_CLASSES, ignored_fraction=0.1, seed=seed) for seed in (1, 2)]
    for predictions, labels in batches:
        miou.update(predictions, labels)
    per_batch = [per_image_confusion(predictions, labels, NUM_CLASSES) for predictions, labels in batches]
    intersection, union, _ = (torch.cat(counts) for counts in zip(*per_batch, strict=True))
    assert torch.equal(miou.intersection, intersection.sum(dim=0))
    assert torch.equal(miou.union, union.sum(dim=0))
    present = miou.union > 0
    assert miou.compute() == pytest.approx((miou.intersection[present] / miou.union[present]).mean().item())


def test_ignored_pixels_count_nowhere() -> None:
    predictions = torch.zeros(2, 8, 8, dtype=torch.long)
    labels = torch.full((2, 8, 8), IGNORE_LABEL)
    assert all(counts.sum() == 0 for counts in per_image_confusion(predictions, labels, NUM_CLASSES))
    miou = MeanIoU(device=torch.device("cpu"))
    miou.update(predictions, labels)
    with pytest.raises(AssertionError, match="no labeled pixels"):
        miou.compute()
