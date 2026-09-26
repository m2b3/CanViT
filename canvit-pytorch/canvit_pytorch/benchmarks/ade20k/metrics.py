"""Segmentation metrics: mean IoU over the dataset, and per-image confusion counts."""

import torch
from torch import Tensor

from canvit_pytorch.benchmarks.ade20k.classes import IGNORE_LABEL, NUM_CLASSES


class MeanIoU:
    """Dataset-level mean IoU: intersections and unions summed over all images, then averaged over classes.

    Accumulates on the device without synchronizing until compute().
    """

    def __init__(self, *, device: torch.device, num_classes: int = NUM_CLASSES, ignore_label: int = IGNORE_LABEL) -> None:
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.intersection = torch.zeros(num_classes, device=device)
        self.union = torch.zeros(num_classes, device=device)

    def update(self, predictions: Tensor, labels: Tensor) -> None:
        """Add [B, H, W] predicted and true labels."""
        assert predictions.ndim == 3 and predictions.shape == labels.shape, (predictions.shape, labels.shape)
        n = self.num_classes
        predicted, true = predictions.flatten().long(), labels.flatten().long()
        keep = true != self.ignore_label
        confusion = torch.bincount(true[keep] * n + predicted[keep], minlength=n * n).view(n, n)
        diagonal = confusion.diag()
        self.intersection += diagonal
        self.union += confusion.sum(dim=1) + confusion.sum(dim=0) - diagonal

    def reset(self) -> None:
        self.intersection.zero_()
        self.union.zero_()

    def compute(self) -> float:
        """Mean over the classes that occur in the labels or the predictions."""
        present = self.union > 0
        assert present.any(), "no labeled pixels were accumulated"
        return (self.intersection / (self.union + 1e-8))[present].mean().item()


def per_image_confusion(predictions: Tensor, labels: Tensor, num_classes: int) -> tuple[Tensor, Tensor, Tensor]:
    """Per image and class: (intersection, union, labeled area) in pixels, three [B, num_classes] float32 tensors.

    One scatter_add over the batch. Counts stay exact in float32 up to 2^24
    pixels per image, so the result does not depend on summation order.
    """
    B = predictions.shape[0]
    assert labels.shape == predictions.shape and predictions.dtype == labels.dtype == torch.int64, (
        predictions.shape, labels.shape, predictions.dtype, labels.dtype,
    )
    pairs = num_classes * num_classes
    pair = predictions * num_classes + labels
    keep = (labels != IGNORE_LABEL) & (pair >= 0) & (pair < pairs)
    image = torch.arange(B, device=predictions.device).view(B, 1, 1).expand_as(predictions)
    flat = (image * pairs + pair)[keep]
    confusion = torch.zeros(B * pairs, dtype=torch.float32, device=predictions.device)
    confusion.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
    confusion = confusion.view(B, num_classes, num_classes)  # [image, predicted, true]
    intersection = confusion.diagonal(dim1=1, dim2=2)
    label_area = confusion.sum(dim=1)
    return intersection, confusion.sum(dim=2) + label_area - intersection, label_area
