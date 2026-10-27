"""FeatureFusion: the feature groups of rl.features merged into a single map."""

import functools
import operator

import torch
from torch import Tensor, nn

from canvit_pytorch.rl.scorer.convnext import LayerNorm2d


class FeatureFusion(nn.Module):
    """[B, sum(group_channels), H, W] -> [B, width, H, W]: BatchNorm over all channels; per group a 1×1 projection
    to the width, an affine LayerNorm and a learned gate; the sum of the groups, a pre-norm residual MLP on each
    cell and a 1×1 projection.

    A group's contribution does not grow with its channel count: a canvas-wide group and an entropy map are
    projected to the same width and normalized alike.
    """

    def __init__(self, group_channels: list[int], width: int) -> None:
        super().__init__()
        self.group_channels = group_channels
        self.input_norm = nn.BatchNorm2d(sum(group_channels))
        self.group_projections = nn.ModuleList([
            nn.Conv2d(channels, width, kernel_size=1) for channels in group_channels
        ])
        self.group_norms = nn.ModuleList([LayerNorm2d(width) for _ in group_channels])
        self.group_gates = nn.Parameter(torch.ones(len(group_channels)))
        self.mlp = nn.Sequential(
            LayerNorm2d(width), nn.Conv2d(width, 2 * width, kernel_size=1), nn.GELU(),
            nn.Conv2d(2 * width, width, kernel_size=1),
        )
        self.output_projection = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, features: Tensor) -> Tensor:
        groups = self.input_norm(features).split(self.group_channels, dim=1)
        parts = zip(self.group_projections, self.group_norms, groups, self.group_gates, strict=True)
        gated = [norm(projection(group)) * gate for projection, norm, group, gate in parts]
        fused = functools.reduce(operator.add, gated)
        return self.output_projection(fused + self.mlp(fused))
