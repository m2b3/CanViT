"""Evaluation metrics: IoU accumulation for segmentation."""

import torch
from torch import Tensor


class mIoUAccumulator:
    """Global mIoU: sum intersection/union across all images, then average over classes.

    No GPU sync until compute().
    """

    def __init__(self, num_classes: int, ignore_index: int, device: torch.device) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.intersection = torch.zeros(num_classes, device=device)
        self.union = torch.zeros(num_classes, device=device)

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """Accumulate from [B, H, W] predictions and targets via confusion matrix."""
        assert preds.ndim == 3, f"Expected [B, H, W], got shape {preds.shape}"
        assert preds.shape == targets.shape, f"Shape mismatch: {preds.shape} vs {targets.shape}"
        n = self.num_classes
        p = preds.flatten().long()
        t = targets.flatten().long()
        valid = t != self.ignore_index
        p, t = p[valid], t[valid]
        # Encode (t, p) pairs into a single integer index for bincount.
        cm = torch.bincount(t * n + p, minlength=n * n).view(n, n)
        diag = cm.diag()
        self.intersection += diag
        self.union += cm.sum(dim=1) + cm.sum(dim=0) - diag

    def reset(self) -> None:
        """Zero all counters (call between validation epochs)."""
        self.intersection.zero_()
        self.union.zero_()

    def compute(self) -> float:
        """Global mIoU. GPU sync happens here."""
        iou = self.intersection / (self.union + 1e-8)
        valid = self.union > 0
        return iou[valid].mean().item() if valid.any() else 0.0
