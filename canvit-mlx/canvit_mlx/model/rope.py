import math
from typing import NamedTuple

import mlx.core as mx


class RoPE(NamedTuple):
    sin: mx.array  # [B, heads=1, patches, head_dim]
    cos: mx.array


def compute_2d_rope(*, positions: mx.array, head_dim: int, base: float) -> RoPE:
    assert head_dim % 4 == 0 and positions.dtype == mx.float32
    frequencies = head_dim // 4
    periods = base ** (mx.arange(frequencies, dtype=mx.float32) / frequencies)
    angles = 2 * math.pi * positions[..., None] / periods
    angles = angles.reshape(*positions.shape[:2], -1)
    angles = mx.concatenate([angles, angles], axis=-1)[:, None]
    return RoPE(sin=mx.sin(angles), cos=mx.cos(angles))


def apply_2d_rope(x: mx.array, rope: RoPE) -> mx.array:
    prefix = x.shape[2] - rope.sin.shape[2]
    assert prefix >= 0, (x.shape, rope.sin.shape)
    spatial = x[:, :, prefix:].astype(mx.float32)
    first, second = mx.split(spatial, 2, axis=-1)
    rotated = spatial * rope.cos + mx.concatenate([-second, first], axis=-1) * rope.sin
    return mx.concatenate([x[:, :, :prefix], rotated.astype(x.dtype)], axis=2)
