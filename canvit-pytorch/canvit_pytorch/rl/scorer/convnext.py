"""ConvNeXt-V2 building blocks (Woo et al., 2023) on channels-first maps [B, C, H, W]."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channels of [B, C, H, W]."""

    def __init__(self, num_channels: int) -> None:
        super().__init__(num_channels, eps=1e-6)

    def forward(self, input: Tensor) -> Tensor:
        channels_last = input.permute(0, 2, 3, 1)
        return F.layer_norm(channels_last, self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)


class GlobalResponseNorm(nn.Module):
    """Global response normalization of channels-last [B, H, W, C]: each channel's spatial L2 norm, relative to
    the mean over channels, rescales it."""

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: Tensor) -> Tensor:
        spatial_norm = x.norm(p=2, dim=(1, 2), keepdim=True)
        relative_norm = spatial_norm / (spatial_norm.mean(dim=-1, keepdim=True) + 1e-6)
        return x + torch.addcmul(self.bias, self.weight, x * relative_norm)


class ConvNeXtV2Block(nn.Module):
    """7×7 depthwise convolution, LayerNorm, an MLP of 4× expansion with global response normalization, a layer
    scale initialized to 1e-6, and the residual connection."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.global_response_norm = GlobalResponseNorm(4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)
        self.layer_scale = nn.Parameter(torch.full((dim,), 1e-6))

    def forward(self, x: Tensor) -> Tensor:
        branch = self.norm(self.depthwise_conv(x).permute(0, 2, 3, 1))
        branch = self.fc2(self.global_response_norm(F.gelu(self.fc1(branch)))).permute(0, 3, 1, 2)
        return branch * self.layer_scale.view(1, -1, 1, 1) + x


class ConvNeXtV2Stage(nn.Module):
    """A 1×1 projection to width when the input has other channels, then num_blocks blocks."""

    def __init__(self, in_channels: int, width: int, num_blocks: int) -> None:
        super().__init__()
        self.projection = nn.Identity() if in_channels == width else nn.Conv2d(in_channels, width, kernel_size=1)
        self.blocks = nn.Sequential(*(ConvNeXtV2Block(width) for _ in range(num_blocks)))

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(self.projection(x))
