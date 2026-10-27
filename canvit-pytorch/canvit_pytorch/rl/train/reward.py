"""What a glimpse is worth: the fraction of the probe's remaining cross-entropy it removes."""

import torch
import torch.nn.functional as F
from torch import Tensor

from canvit_pytorch.benchmarks.ade20k import IGNORE_LABEL


def segmentation_cross_entropy(logits: Tensor, labels: Tensor, *, resolution_px: int) -> Tensor:
    """[B]: each scene's mean cross-entropy over its labeled pixels.

    The logits [B, num_classes, G, G] are bilinearly upsampled to resolution_px, the labels [B, S, S] subsampled
    to it; S must be a multiple of resolution_px.
    """
    stride, remainder = divmod(labels.shape[-1], resolution_px)
    assert remainder == 0, f"{resolution_px} px does not divide the {labels.shape[-1]} px labels"
    labels = labels[:, ::stride, ::stride]
    upsampled = F.interpolate(logits, size=(resolution_px, resolution_px), mode="bilinear", align_corners=False)
    loss = F.cross_entropy(upsampled, labels, ignore_index=IGNORE_LABEL, reduction="none")
    return loss.sum(dim=(1, 2)) / (labels != IGNORE_LABEL).sum(dim=(1, 2)).clamp(min=1)


def fraction_removed(loss_before: Tensor, loss_after: Tensor) -> Tensor:
    """A glimpse's reward: (before - after) / before, elementwise."""
    return (loss_before - loss_after) / loss_before.clamp_min(1e-4)


class RunningStandardizer:
    """Standardizes batches of parallel scalar streams [num_streams, N] by exponential moving averages of each
    stream's mean and mean square, bias-corrected as in Adam so that the first batches are already unit-scale.

    Statistics are shared across the N samples of a batch, never per sample, so that a single sampled viewpoint
    per scene is a valid regression target.
    """

    def __init__(self, *, num_streams: int, momentum: float, device: torch.device) -> None:
        self.momentum = momentum
        self.mean = torch.zeros(num_streams, 1, device=device)
        self.mean_square = torch.zeros(num_streams, 1, device=device)
        self.num_updates = 0

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Update the statistics with x [num_streams, N], then standardize x."""
        self.num_updates += 1
        self.mean = self.momentum * self.mean + (1 - self.momentum) * x.mean(dim=1, keepdim=True)
        self.mean_square = self.momentum * self.mean_square + (1 - self.momentum) * (x * x).mean(dim=1, keepdim=True)
        correction = 1 - self.momentum**self.num_updates
        mean, mean_square = self.mean / correction, self.mean_square / correction
        return (x - mean) / (mean_square - mean**2).clamp_min(1e-4).sqrt()
