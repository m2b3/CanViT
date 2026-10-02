"""Scene-relative two-dimensional rotary position embeddings."""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array


class RoPE(NamedTuple):
    sin: Array
    cos: Array


def make_rope_periods(*, head_dim: int, base: float) -> Array:
    assert head_dim % 4 == 0, f"2D RoPE needs head_dim divisible by 4, got {head_dim}"
    assert base > 0, base
    frequencies = head_dim // 4
    return base ** (jnp.arange(frequencies, dtype=jnp.float32) / frequencies)


def compute_2d_rope(*, positions: Array, periods: Array) -> RoPE:
    assert positions.dtype == periods.dtype == jnp.float32, (positions.dtype, periods.dtype)
    assert positions.ndim == 3 and positions.shape[-1] == 2, positions.shape
    angles = 2.0 * math.pi * positions[..., None] / periods
    angles = jnp.tile(angles.reshape(positions.shape[0], positions.shape[1], -1), (1, 1, 2))
    return RoPE(sin=jnp.sin(angles)[:, :, None, :], cos=jnp.cos(angles)[:, :, None, :])


def apply_2d_rope(x: Array, rope: RoPE) -> Array:
    """Rotate spatial tokens in ``[batch, tokens, heads, head_dim]`` arrays."""
    assert x.ndim == 4, x.shape
    assert rope.sin.dtype == rope.cos.dtype == jnp.float32, (rope.sin.dtype, rope.cos.dtype)
    assert rope.sin.shape == rope.cos.shape, (rope.sin.shape, rope.cos.shape)
    prefix = x.shape[1] - rope.sin.shape[1]
    assert prefix >= 0, (x.shape, rope.sin.shape)
    assert rope.sin.shape[-1] == x.shape[-1], (x.shape, rope.sin.shape)
    spatial = x[:, prefix:].astype(jnp.float32)
    half = spatial.shape[-1] // 2
    first, second = spatial[..., :half], spatial[..., half:]
    rotated = spatial * rope.cos + jnp.concatenate((-second, first), axis=-1) * rope.sin
    if prefix == 0:
        return rotated.astype(x.dtype)
    return jnp.concatenate((x[:, :prefix], rotated.astype(x.dtype)), axis=1)
