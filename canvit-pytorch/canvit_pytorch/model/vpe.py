"""Viewpoint Encoding (VPE): the viewpoint as one glimpse-stream token.

(row, col, scale) becomes (row/scale, col/scale, log scale), whose Euclidean
distances are invariant to rescaling the scene (paper, Appendix B), then random
Fourier features and a LayerNorm lift it to the backbone width.
"""

import math

import torch
from torch import Tensor, nn

from canvit_pytorch.viewpoint import Viewpoint


class ViewpointEncoding(nn.Module):
    frequencies: Tensor  # [dim // 2, 3], fixed at construction

    def __init__(self, dim: int, seed: int = 42) -> None:
        super().__init__()
        assert dim > 0 and dim % 2 == 0, dim
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer("frequencies", torch.randn(dim // 2, 3, generator=generator, dtype=torch.float32))
        self.norm = nn.LayerNorm(dim)
        self.norm.weight.data.fill_(1.0 / math.sqrt(dim))

    def forward(self, viewpoint: Viewpoint) -> Tensor:
        """[B, dim], float32 whatever the autocast context."""
        row, col, scale = viewpoint.centers[:, 0], viewpoint.centers[:, 1], viewpoint.scales
        with torch.autocast(device_type=scale.device.type, enabled=False):
            z = torch.stack([row / scale, col / scale, torch.log(scale)], dim=-1)
            phases = z @ self.frequencies.T
            return self.norm(torch.cat([torch.cos(phases), torch.sin(phases)], dim=-1))
