"""What a ViewpointScorer reads: the canvas and the probe's segmentation of it, compared with earlier canvases.

At each canvas cell, in channel order:

- the entropy of the probe's class distribution, in nats, and its change since the previous canvas;
- the cosine dissimilarity of the layer-normalized canvas patch to the same patch of the previous canvas, and to
  that of the canvas before the first glimpse;
- the layer-normalized canvas patch, and its change since the previous canvas.

The previous canvas is the one of the previous decision, or the canvas before the first glimpse.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from canvit_pytorch.model.segmentation import CanViTForSemanticSegmentation
from canvit_pytorch.policies.entropy import predictive_entropy

FEATURE_GRID_SIZE = 32
"""Features of any canvas grid are average-pooled to this grid, ViewpointScorer's input."""


def feature_channels(canvas_dim: int) -> list[int]:
    """The channels of each feature group, in channel order."""
    return [1, 1, 1, 1, canvas_dim, canvas_dim]


@dataclass(frozen=True)
class CanvasReading:
    logits: Tensor  # [B, num_classes, G, G] the probe's segmentation of the canvas, float32
    entropy: Tensor  # [B, G, G] of the probe's class distribution at each canvas cell, in nats
    patches: Tensor  # [B, canvas_dim, G, G] layer-normalized canvas patches, float32


def read_canvas(model: CanViTForSemanticSegmentation, canvas: Tensor) -> CanvasReading:
    """The probe's segmentation of a canvas and its layer-normalized patches, in float32 also under autocast."""
    with torch.autocast(canvas.device.type, enabled=False):
        logits = model.logits(canvas)
        patches = model.canvit.canvas_patch_grid(canvas).float()
        normalized = F.layer_norm(patches, (patches.shape[-1],)).permute(0, 3, 1, 2)
        return CanvasReading(logits=logits, entropy=predictive_entropy(logits), patches=normalized)


class EpisodeFeatures:
    """The scorer's input at each decision of an episode, from the readings of its canvas so far."""

    def __init__(self, initial: CanvasReading) -> None:
        """initial: the reading before the first glimpse."""
        self.initial = initial
        self.previous = initial

    def __call__(self, current: CanvasReading) -> Tensor:
        """[B, sum(feature_channels), FEATURE_GRID_SIZE, FEATURE_GRID_SIZE] at the decision on `current`, which
        becomes the previous reading."""
        previous, self.previous = self.previous, current
        features = torch.cat([
            current.entropy.unsqueeze(1),
            (current.entropy - previous.entropy).unsqueeze(1),
            (1 - F.cosine_similarity(current.patches, previous.patches, dim=1)).unsqueeze(1),
            (1 - F.cosine_similarity(current.patches, self.initial.patches, dim=1)).unsqueeze(1),
            current.patches,
            current.patches - previous.patches,
        ], dim=1)
        return F.adaptive_avg_pool2d(features, FEATURE_GRID_SIZE)
