"""DINOv3-style glimpse backbone with scene-relative RoPE."""

import math

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from canvit_core.backbone import ViTSpec

from canvit_nnx.model.initialization import DINO_LINEAR_KERNEL, symmetric_uniform
from canvit_nnx.model.rope import RoPE, apply_2d_rope

Array = jax.Array


class PatchEmbed(nnx.Module):
    def __init__(self, *, patch_size: int, embed_dim: int, rngs: nnx.Rngs) -> None:
        bound = math.sqrt(1.0 / (3 * patch_size**2))
        self.proj = nnx.Conv(
            in_features=3,
            out_features=embed_dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            kernel_init=symmetric_uniform(bound=bound),
            bias_init=symmetric_uniform(bound=bound),
            rngs=rngs,
        )

    def __call__(self, x: Array) -> Array:
        embedded = self.proj(x)
        batch_size, height, width, dim = embedded.shape
        return embedded.reshape(batch_size, height * width, dim)


class LayerScale(nnx.Module):
    def __init__(self, *, dim: int, init_value: float) -> None:
        self.gamma = nnx.Param(jnp.full((dim,), init_value, dtype=jnp.float32))

    def __call__(self, x: Array) -> Array:
        return x * self.gamma[...]


class MLP(nnx.Module):
    def __init__(self, *, dim: int, hidden_dim: int, rngs: nnx.Rngs) -> None:
        self.fc1 = nnx.Linear(
            in_features=dim,
            out_features=hidden_dim,
            kernel_init=DINO_LINEAR_KERNEL,
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs,
        )
        self.fc2 = nnx.Linear(
            in_features=hidden_dim,
            out_features=dim,
            kernel_init=DINO_LINEAR_KERNEL,
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs,
        )

    def __call__(self, x: Array) -> Array:
        return self.fc2(jax.nn.gelu(self.fc1(x), approximate=False))


class SelfAttention(nnx.Module):
    def __init__(self, *, dim: int, num_heads: int, rngs: nnx.Rngs) -> None:
        assert dim % num_heads == 0, (dim, num_heads)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nnx.Linear(
            in_features=dim,
            out_features=3 * dim,
            kernel_init=DINO_LINEAR_KERNEL,
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs,
        )
        self.proj = nnx.Linear(
            in_features=dim,
            out_features=dim,
            kernel_init=DINO_LINEAR_KERNEL,
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs,
        )

    def __call__(self, x: Array, rope: RoPE) -> Array:
        batch_size, tokens, dim = x.shape
        assert self.qkv.bias is not None
        bias_mask = jnp.concatenate(
            (
                jnp.ones((dim,), dtype=jnp.float32),
                jnp.zeros((dim,), dtype=jnp.float32),
                jnp.ones((dim,), dtype=jnp.float32),
            )
        )
        qkv = x @ self.qkv.kernel[...] + self.qkv.bias[...] * bias_mask
        qkv = qkv.reshape(batch_size, tokens, 3, self.num_heads, self.head_dim)
        query, key, value = (qkv[:, :, index] for index in range(3))
        query = apply_2d_rope(query, rope)
        key = apply_2d_rope(key, rope)
        output = jax.nn.dot_product_attention(
            query,
            key,
            value,
            scale=self.head_dim**-0.5,
        )
        return self.proj(output.reshape(batch_size, tokens, dim))


class ViTBlock(nnx.Module):
    def __init__(self, *, spec: ViTSpec, rngs: nnx.Rngs) -> None:
        dim = spec.embed_dim
        self.norm1 = nnx.LayerNorm(dim, epsilon=1e-5, use_fast_variance=False, rngs=rngs)
        self.attn = SelfAttention(dim=dim, num_heads=spec.num_heads, rngs=rngs)
        self.ls1 = LayerScale(dim=dim, init_value=spec.layerscale_init)
        self.norm2 = nnx.LayerNorm(dim, epsilon=1e-5, use_fast_variance=False, rngs=rngs)
        self.mlp = MLP(dim=dim, hidden_dim=spec.mlp_hidden_dim, rngs=rngs)
        self.ls2 = LayerScale(dim=dim, init_value=spec.layerscale_init)

    def __call__(self, x: Array, rope: RoPE) -> Array:
        x = x + self.ls1(self.attn(self.norm1(x), rope))
        return x + self.ls2(self.mlp(self.norm2(x)))


class ViTBackbone(nnx.Module):
    def __init__(self, *, spec: ViTSpec, rngs: nnx.Rngs) -> None:
        self.spec = spec
        self.patch_embed = PatchEmbed(patch_size=spec.patch_size, embed_dim=spec.embed_dim, rngs=rngs)
        self.blocks = nnx.List([ViTBlock(spec=spec, rngs=rngs) for _ in range(spec.num_blocks)])


__all__ = ["PatchEmbed", "ViTBackbone", "ViTBlock", "ViTSpec"]
