"""Canvas Attention: glimpse tokens read from the canvas, and the canvas is written from glimpse tokens.

Both inputs are layer-normalized and rotated with SR-RoPE in float32. Queries,
keys, values and attention probabilities enter the matrix products in the
compute dtype; the products accumulate in float32. Projections are learned on
the glimpse side; the canvas side has them only with "qkvo" projections.
Without them, bfloat16 Canvas Attention on TPU runs as the Pallas kernels of
fused_canvas_attention, which compute the same function.
"""

from typing import assert_never

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from canvit_core.config import CanvasProjections

from canvit_nnx.model import fused_canvas_attention
from canvit_nnx.model.initialization import pytorch_linear_initializers
from canvit_nnx.model.multihead import attention, from_multihead, to_multihead
from canvit_nnx.model.rope import RoPE, apply_2d_rope

Array = jax.Array


class Identity(nnx.Module):
    def __call__(self, x: Array) -> Array:
        return x


def _linear(*, in_dim: int, out_dim: int, dtype: jnp.dtype, rngs: nnx.Rngs) -> nnx.Linear:
    kernel_init, bias_init = pytorch_linear_initializers(in_features=in_dim)
    return nnx.Linear(
        in_features=in_dim,
        out_features=out_dim,
        kernel_init=kernel_init,
        bias_init=bias_init,
        dtype=dtype,
        rngs=rngs,
    )


def _layer_norm(dim: int, *, rngs: nnx.Rngs) -> nnx.LayerNorm:
    return nnx.LayerNorm(dim, epsilon=1e-5, use_fast_variance=False, rngs=rngs)


def _canvas_side_map(
    *, projections: CanvasProjections, canvas_dim: int, dtype: jnp.dtype, rngs: nnx.Rngs
) -> nnx.Linear | Identity:
    match projections:
        case "asymmetric":
            return Identity()
        case "qkvo":
            return _linear(in_dim=canvas_dim, out_dim=canvas_dim, dtype=dtype, rngs=rngs)
        case _:
            assert_never(projections)


class CanvasAttentionRead(nnx.Module):
    """Glimpse tokens query the canvas; returns the updated glimpse tokens."""

    def __init__(
        self,
        *,
        backbone_dim: int,
        canvas_dim: int,
        num_heads: int,
        projections: CanvasProjections,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ) -> None:
        assert canvas_dim % num_heads == 0, (canvas_dim, num_heads)
        self.num_heads = num_heads
        self.projections: CanvasProjections = projections
        self.dtype = dtype
        self.ln_q = _layer_norm(backbone_dim, rngs=rngs)
        self.ln_kv = _layer_norm(canvas_dim, rngs=rngs)
        self.q_map = _linear(in_dim=backbone_dim, out_dim=canvas_dim, dtype=dtype, rngs=rngs)
        self.k_map = _canvas_side_map(projections=projections, canvas_dim=canvas_dim, dtype=dtype, rngs=rngs)
        self.v_map = _canvas_side_map(projections=projections, canvas_dim=canvas_dim, dtype=dtype, rngs=rngs)
        self.o_map = _linear(in_dim=canvas_dim, out_dim=backbone_dim, dtype=dtype, rngs=rngs)

    def __call__(self, *, tokens: Array, canvas: Array, tokens_rope: RoPE, canvas_rope: RoPE) -> Array:
        queries = apply_2d_rope(to_multihead(self.q_map(self.ln_q(tokens)), self.num_heads), tokens_rope)
        if self.projections == "asymmetric" and fused_canvas_attention.use_fused_kernels(self.dtype):
            assert self.ln_kv.scale is not None and self.ln_kv.bias is not None
            heads = fused_canvas_attention.fused_canvas_read(
                canvas=canvas,
                queries=queries,
                ln_scale=self.ln_kv.scale[...],
                ln_bias=self.ln_kv.bias[...],
                epsilon=self.ln_kv.epsilon,
                canvas_rope=canvas_rope,
            )
        else:
            canvas_normalized = self.ln_kv(canvas)
            keys = apply_2d_rope(to_multihead(self.k_map(canvas_normalized), self.num_heads), canvas_rope)
            values = to_multihead(self.v_map(canvas_normalized), self.num_heads)
            heads = attention(queries, keys.astype(self.dtype), values.astype(self.dtype), probability_dtype=self.dtype)
        return tokens + self.o_map(from_multihead(heads))


class CanvasAttentionWrite(nnx.Module):
    """The canvas queries glimpse tokens; returns the updated canvas.

    The update is added to the canvas in float32 and rounded once to the
    canvas dtype.
    """

    def __init__(
        self,
        *,
        backbone_dim: int,
        canvas_dim: int,
        num_heads: int,
        projections: CanvasProjections,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ) -> None:
        assert canvas_dim % num_heads == 0, (canvas_dim, num_heads)
        self.num_heads = num_heads
        self.projections: CanvasProjections = projections
        self.dtype = dtype
        self.ln_q = _layer_norm(canvas_dim, rngs=rngs)
        self.ln_kv = _layer_norm(backbone_dim, rngs=rngs)
        self.q_map = _canvas_side_map(projections=projections, canvas_dim=canvas_dim, dtype=dtype, rngs=rngs)
        self.k_map = _linear(in_dim=backbone_dim, out_dim=canvas_dim, dtype=dtype, rngs=rngs)
        self.v_map = _linear(in_dim=backbone_dim, out_dim=canvas_dim, dtype=dtype, rngs=rngs)
        self.o_map = _canvas_side_map(projections=projections, canvas_dim=canvas_dim, dtype=dtype, rngs=rngs)

    def _keys_values(self, tokens: Array, tokens_rope: RoPE) -> tuple[Array, Array]:
        assert self.k_map.bias is not None and self.v_map.bias is not None
        # One matrix product for both projections: XLA does not merge the two on its own.
        kernel = jnp.concatenate((self.k_map.kernel[...], self.v_map.kernel[...]), axis=1)
        bias = jnp.concatenate((self.k_map.bias[...], self.v_map.bias[...]))
        projected = self.ln_kv(tokens).astype(self.dtype) @ kernel.astype(self.dtype) + bias.astype(self.dtype)
        keys, values = jnp.split(projected, 2, axis=-1)
        return apply_2d_rope(to_multihead(keys, self.num_heads), tokens_rope), to_multihead(values, self.num_heads)

    def __call__(self, *, canvas: Array, tokens: Array, canvas_rope: RoPE, tokens_rope: RoPE) -> Array:
        keys, values = self._keys_values(tokens, tokens_rope)
        if self.projections == "asymmetric" and fused_canvas_attention.use_fused_kernels(self.dtype):
            assert self.ln_q.scale is not None and self.ln_q.bias is not None
            return fused_canvas_attention.fused_canvas_write(
                canvas=canvas,
                keys=keys,
                values=values,
                ln_scale=self.ln_q.scale[...],
                ln_bias=self.ln_q.bias[...],
                epsilon=self.ln_q.epsilon,
                canvas_rope=canvas_rope,
            )
        queries = apply_2d_rope(to_multihead(self.q_map(self.ln_q(canvas)), self.num_heads), canvas_rope)
        heads = attention(queries.astype(self.dtype), keys, values, probability_dtype=self.dtype)
        update = self.o_map(from_multihead(heads))
        return (canvas.astype(jnp.float32) + update.astype(jnp.float32)).astype(canvas.dtype)


__all__ = ["CanvasAttentionRead", "CanvasAttentionWrite"]
