"""Viewpoint Encoding for the glimpse token stream."""

import math

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from canvit_nnx.viewpoint import Viewpoint

Array = jax.Array


class ViewpointEncoding(nnx.Module):
    def __init__(self, *, dim: int, rngs: nnx.Rngs, seed: int = 42) -> None:
        assert dim > 0 and dim % 2 == 0, dim
        self.frequencies = nnx.Variable(jax.random.normal(jax.random.PRNGKey(seed), (dim // 2, 3), dtype=jnp.float32))
        self.norm = nnx.LayerNorm(dim, epsilon=1e-5, use_fast_variance=False, rngs=rngs)
        assert self.norm.scale is not None
        self.norm.scale[...] = jnp.full((dim,), 1.0 / math.sqrt(dim), dtype=jnp.float32)

    def __call__(self, viewpoint: Viewpoint) -> Array:
        """[B, dim] in float32, as canvit-pytorch computes it outside autocast."""
        scale = viewpoint.scales[:, None]
        coordinates = jnp.concatenate((viewpoint.centers / scale, jnp.log(scale)), axis=-1)
        # Phases reach tens of radians, so the product must be float32: by default a TPU rounds float32 matmul
        # operands to bfloat16, and a GPU may use TF32.
        phases = jnp.matmul(coordinates, self.frequencies[...].T, precision=jax.lax.Precision.HIGHEST)
        return self.norm(jnp.concatenate((jnp.cos(phases), jnp.sin(phases)), axis=-1))
