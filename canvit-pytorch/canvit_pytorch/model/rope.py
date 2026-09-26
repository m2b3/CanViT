"""Scene-relative 2D rotary position embeddings (SR-RoPE).

Glimpse patches and canvas patches get their rotations from their positions in
the scene frame, so attention between the two streams is spatially aligned.
Computed in float32 throughout.
"""

import math
from typing import NamedTuple

import torch
from torch import Tensor


class RoPE(NamedTuple):
    sin: Tensor  # [B, 1, N, head_dim]
    cos: Tensor  # [B, 1, N, head_dim]


def make_rope_periods(*, head_dim: int, base: float, device: torch.device) -> Tensor:
    """Wavelengths in scene units, geometric from 1 to base: [head_dim // 4], float32."""
    assert head_dim % 4 == 0, f"2D RoPE needs head_dim divisible by 4, got {head_dim}"
    n_freqs = head_dim // 4
    return base ** (torch.arange(n_freqs, device=device, dtype=torch.float32) / n_freqs)


def compute_2d_rope(*, positions: Tensor, periods: Tensor) -> RoPE:
    """Rotations for [B, N, 2] float32 scene positions."""
    assert positions.dtype == periods.dtype == torch.float32, (positions.dtype, periods.dtype)
    assert positions.ndim == 3 and positions.shape[-1] == 2, positions.shape
    angles = 2 * math.pi * positions.unsqueeze(-1) / periods  # [B, N, 2, n_freqs]
    angles = angles.flatten(-2, -1).tile((2,))  # [B, N, head_dim]
    return RoPE(sin=torch.sin(angles).unsqueeze(1), cos=torch.cos(angles).unsqueeze(1))


def apply_2d_rope(x: Tensor, rope: RoPE) -> Tensor:
    """Rotate the last N tokens of [B, heads, N_total, head_dim]; earlier tokens pass through.

    The unrotated prefix holds the tokens without a position (VPE, CLS, registers).
    """
    assert x.ndim == 4, x.shape
    assert rope.sin.dtype == torch.float32, rope.sin.dtype
    n_prefix = x.shape[2] - rope.sin.shape[2]
    assert n_prefix >= 0, f"RoPE covers {rope.sin.shape[2]} tokens, input has {x.shape[2]}"
    half = x.shape[3] // 2
    out = torch.empty_like(x)
    out[:, :, :n_prefix] = x[:, :, :n_prefix]
    rotated = x[:, :, n_prefix:].float()
    swapped = torch.cat([-rotated[..., half:], rotated[..., :half]], dim=-1)
    out[:, :, n_prefix:] = (rotated * rope.cos + swapped * rope.sin).to(x.dtype)
    return out
