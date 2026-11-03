"""Multi-head attention shared by the backbone and Canvas Attention."""

import jax
import jax.numpy as jnp

Array = jax.Array


def to_multihead(x: Array, num_heads: int) -> Array:
    """[B, N, num_heads * head_dim] -> [B, N, num_heads, head_dim]."""
    batch_size, tokens, dim = x.shape
    assert dim % num_heads == 0, (dim, num_heads)
    return x.reshape(batch_size, tokens, num_heads, dim // num_heads)


def from_multihead(x: Array) -> Array:
    """[B, N, num_heads, head_dim] -> [B, N, num_heads * head_dim]."""
    batch_size, tokens, num_heads, head_dim = x.shape
    return x.reshape(batch_size, tokens, num_heads * head_dim)


@jax.custom_vjp
def _softmax(scores: Array) -> Array:
    """jax.nn.softmax over the last axis, whose score cotangent is computed apart from the products that consume it.

    The score cotangent p * (dp - sum(dp * p)) nearly cancels when the values
    of every key share a component that is large against their differences,
    as in trained Canvas Attention Writes. Fused into the matrix products
    that consume it, with dp rounded to bfloat16, XLA on TPU computed it with
    errors larger than the result; behind an optimization barrier it is a
    float32 array of its own.
    """
    return jax.nn.softmax(scores, axis=-1)


def _softmax_forward(scores: Array) -> tuple[Array, Array]:
    probabilities = jax.nn.softmax(scores, axis=-1)
    return probabilities, probabilities


def _softmax_backward(probabilities: Array, cotangent: Array) -> tuple[Array]:
    weighted = jnp.sum(cotangent * probabilities, axis=-1, keepdims=True)
    return (jax.lax.optimization_barrier(probabilities * (cotangent - weighted)),)


_softmax.defvjp(_softmax_forward, _softmax_backward)


def attention(queries: Array, keys: Array, values: Array, *, probability_dtype: jnp.dtype) -> Array:
    """Scaled dot-product attention of [B, Nq, H, d] queries over [B, Nk, H, d] keys and values.

    Scores, softmax and the weighted sum of values accumulate in float32; the
    probabilities are rounded to probability_dtype before the sum. Returns
    float32 [B, Nq, H, d].
    """
    scores = jnp.einsum("bqhd,bkhd->bhqk", queries, keys, preferred_element_type=jnp.float32)
    probabilities = _softmax(scores * queries.shape[-1] ** -0.5).astype(probability_dtype)
    return jnp.einsum("bhqk,bkhd->bqhd", probabilities, values, preferred_element_type=jnp.float32)


__all__ = ["attention", "from_multihead", "to_multihead"]
