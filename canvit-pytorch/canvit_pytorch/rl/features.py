"""What a ViewpointScorer reads: the canvas and the probe's segmentation of it, compared with earlier canvases.

A scorer reads a selection of these feature groups at each canvas cell (the published scorers read all of them, in
this order):

- entropy: the entropy of the probe's class distribution, in nats; entropy_change: its change since the previous
  canvas;
- dissimilarity_to_previous, dissimilarity_to_initial: the cosine dissimilarity of the layer-normalized canvas patch
  to the same patch of the previous canvas, and to that of the canvas before the first glimpse;
- patches: the layer-normalized canvas patch; patch_change: its change since the previous canvas.

The previous canvas is the one of the previous decision, or the canvas before the first glimpse.
"""

from dataclasses import dataclass
from typing import Literal, assert_never, get_args

import torch
import torch.nn.functional as F
from torch import Tensor

from canvit_pytorch.model.segmentation import CanViTForSemanticSegmentation
from canvit_pytorch.policies.entropy import predictive_entropy

FEATURE_GRID_SIZE = 32
"""Features of any canvas grid are average-pooled to this grid, ViewpointScorer's input."""

FeatureGroup = Literal[
    "entropy", "entropy_change", "dissimilarity_to_previous", "dissimilarity_to_initial", "patches", "patch_change",
]
FEATURE_GROUPS: tuple[FeatureGroup, ...] = get_args(FeatureGroup)
"""Every group, in the order the published scorers read them."""


def feature_channels(groups: tuple[FeatureGroup, ...], canvas_dim: int) -> list[int]:
    """The channels of each group, in the order of groups."""
    return [canvas_dim if group in ("patches", "patch_change") else 1 for group in groups]


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


def dissimilarity(a: CanvasReading, b: CanvasReading) -> Tensor:
    """[B, 1, G, G]: the cosine dissimilarity of each patch of a to the same patch of b."""
    return (1 - F.cosine_similarity(a.patches, b.patches, dim=1)).unsqueeze(1)


class EpisodeFeatures:
    """A scorer's input at each decision of an episode, from the readings of its canvas so far: the feature groups
    it reads, in their order."""

    def __init__(self, initial: CanvasReading, groups: tuple[FeatureGroup, ...]) -> None:
        """initial: the reading before the first glimpse."""
        self.initial = initial
        self.previous = initial
        self.groups = groups

    def __call__(self, current: CanvasReading) -> Tensor:
        """[B, sum(feature_channels), FEATURE_GRID_SIZE, FEATURE_GRID_SIZE] at the decision on `current`, which
        becomes the previous reading."""
        previous, self.previous = self.previous, current
        features = torch.cat([self.group(group, current, previous) for group in self.groups], dim=1)
        return F.adaptive_avg_pool2d(features, FEATURE_GRID_SIZE)

    def group(self, group: FeatureGroup, current: CanvasReading, previous: CanvasReading) -> Tensor:
        """[B, channels, G, G]: one group at the decision on `current`."""
        match group:
            case "entropy":
                return current.entropy.unsqueeze(1)
            case "entropy_change":
                return (current.entropy - previous.entropy).unsqueeze(1)
            case "dissimilarity_to_previous":
                return dissimilarity(current, previous)
            case "dissimilarity_to_initial":
                return dissimilarity(current, self.initial)
            case "patches":
                return current.patches
            case "patch_change":
                return current.patches - previous.patches
            case _:
                assert_never(group)
