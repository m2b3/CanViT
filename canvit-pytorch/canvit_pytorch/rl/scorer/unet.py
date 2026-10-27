"""A U-Net of ConvNeXt-V2 stages whose encoder pools down to a single cell, so that every output cell depends on the
whole input."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from canvit_pytorch.rl.scorer.convnext import ConvNeXtV2Stage


class UNet(nn.Module):
    """[B, width, G, G] -> [B, width, G, G] for a power of two G: encoder stages at G, G/2, ..., 1 cells, separated by
    2×2 average pooling; decoder stages back up to G, each on the nearest-neighbour upsampling of the previous output
    concatenated with the encoder's map of the same size."""

    def __init__(self, *, grid_size: int, width: int, blocks_per_stage: int) -> None:
        super().__init__()
        assert grid_size > 0 and grid_size & (grid_size - 1) == 0, f"grid_size {grid_size} is not a power of two"
        num_poolings = grid_size.bit_length() - 1
        self.encoder_stages = nn.ModuleList([
            ConvNeXtV2Stage(width, width, blocks_per_stage) for _ in range(num_poolings + 1)
        ])
        self.decoder_stages = nn.ModuleList([
            ConvNeXtV2Stage(2 * width, width, blocks_per_stage) for _ in range(num_poolings)
        ])

    def forward(self, x: Tensor) -> Tensor:
        skips = []
        for i, stage in enumerate(self.encoder_stages):
            x = stage(x if i == 0 else F.avg_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for stage, skip in zip(self.decoder_stages, reversed(skips), strict=True):
            x = stage(torch.cat([F.interpolate(x, scale_factor=2, mode="nearest"), skip], dim=1))
        return x
