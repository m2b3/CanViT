"""Canvas Attention read and write modules."""

import flax.nnx as nnx
import jax
from canvit_core.config import CanvasProjections

from canvit_nnx.model.initialization import pytorch_linear_initializers
from canvit_nnx.model.rope import RoPE, apply_2d_rope

Array = jax.Array


class Identity(nnx.Module):
    def __call__(self, x: Array) -> Array:
        return x


def _to_multihead(x: Array, num_heads: int) -> Array:
    batch_size, tokens, dim = x.shape
    assert dim % num_heads == 0, (dim, num_heads)
    return x.reshape(batch_size, tokens, num_heads, dim // num_heads)


def _from_multihead(x: Array) -> Array:
    batch_size, tokens, num_heads, head_dim = x.shape
    return x.reshape(batch_size, tokens, num_heads * head_dim)


def _linear(*, in_dim: int, out_dim: int, rngs: nnx.Rngs) -> nnx.Linear:
    kernel_init, bias_init = pytorch_linear_initializers(in_features=in_dim)
    return nnx.Linear(
        in_features=in_dim,
        out_features=out_dim,
        kernel_init=kernel_init,
        bias_init=bias_init,
        rngs=rngs,
    )


class CanvasAttention(nnx.Module):
    def __init__(
        self,
        *,
        query_dim: int,
        kv_dim: int,
        canvas_dim: int,
        num_heads: int,
        q_map: nnx.Module,
        k_map: nnx.Module,
        v_map: nnx.Module,
        o_map: nnx.Module,
        rngs: nnx.Rngs,
    ) -> None:
        assert canvas_dim % num_heads == 0, (canvas_dim, num_heads)
        self.num_heads = num_heads
        self.head_dim = canvas_dim // num_heads
        self.q_map = q_map
        self.k_map = k_map
        self.v_map = v_map
        self.o_map = o_map
        self.ln_q = nnx.LayerNorm(query_dim, epsilon=1e-5, use_fast_variance=False, rngs=rngs)
        self.ln_kv = nnx.LayerNorm(kv_dim, epsilon=1e-5, use_fast_variance=False, rngs=rngs)

    def __call__(
        self,
        *,
        query: Array,
        kv: Array,
        query_rope: RoPE,
        kv_rope: RoPE,
    ) -> Array:
        query_heads = _to_multihead(self.q_map(self.ln_q(query)), self.num_heads)
        kv_normalized = self.ln_kv(kv)
        key_heads = _to_multihead(self.k_map(kv_normalized), self.num_heads)
        value_heads = _to_multihead(self.v_map(kv_normalized), self.num_heads)
        query_heads = apply_2d_rope(query_heads, query_rope)
        key_heads = apply_2d_rope(key_heads, kv_rope)
        output = jax.nn.dot_product_attention(
            query_heads,
            key_heads.astype(query_heads.dtype),
            value_heads.astype(query_heads.dtype),
            scale=self.head_dim**-0.5,
        )
        return self.o_map(_from_multihead(output))


class CanvasAttentionRead(CanvasAttention):
    def __init__(
        self,
        *,
        backbone_dim: int,
        canvas_dim: int,
        num_heads: int,
        projections: CanvasProjections,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__(
            query_dim=backbone_dim,
            kv_dim=canvas_dim,
            canvas_dim=canvas_dim,
            num_heads=num_heads,
            q_map=_linear(in_dim=backbone_dim, out_dim=canvas_dim, rngs=rngs),
            k_map=_linear(in_dim=canvas_dim, out_dim=canvas_dim, rngs=rngs) if projections == "qkvo" else Identity(),
            v_map=_linear(in_dim=canvas_dim, out_dim=canvas_dim, rngs=rngs) if projections == "qkvo" else Identity(),
            o_map=_linear(in_dim=canvas_dim, out_dim=backbone_dim, rngs=rngs),
            rngs=rngs,
        )


class CanvasAttentionWrite(CanvasAttention):
    def __init__(
        self,
        *,
        backbone_dim: int,
        canvas_dim: int,
        num_heads: int,
        projections: CanvasProjections,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__(
            query_dim=canvas_dim,
            kv_dim=backbone_dim,
            canvas_dim=canvas_dim,
            num_heads=num_heads,
            q_map=_linear(in_dim=canvas_dim, out_dim=canvas_dim, rngs=rngs) if projections == "qkvo" else Identity(),
            k_map=_linear(in_dim=backbone_dim, out_dim=canvas_dim, rngs=rngs),
            v_map=_linear(in_dim=backbone_dim, out_dim=canvas_dim, rngs=rngs),
            o_map=_linear(in_dim=canvas_dim, out_dim=canvas_dim, rngs=rngs) if projections == "qkvo" else Identity(),
            rngs=rngs,
        )


__all__ = ["CanvasAttention", "CanvasAttentionRead", "CanvasAttentionWrite"]
