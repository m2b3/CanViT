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
        scale = viewpoint.scales[:, None]
        coordinates = jnp.concatenate((viewpoint.centers / scale, jnp.log(scale)), axis=-1)
        phases = coordinates @ self.frequencies[...].T
        return self.norm(jnp.concatenate((jnp.cos(phases), jnp.sin(phases)), axis=-1))
