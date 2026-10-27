"""AdaGlimpse's masked-autoencoder ViT, and its rollout over a growing set of glimpse patches.

AdaGlimpse keeps no recurrent latent: it holds every patch it has seen and
encodes the whole set again after each glimpse, then decodes the encoding
into a feature per cell of its output grid. Parameter names follow the
upstream modules (`MaskedAutoencoderViT`, `Layer_scale_init_Block`), so each
maps to one tensor of an AdaGlimpse state dict.
"""

from typing import NamedTuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from canvit_nnx.viewpoint import Viewpoint
from jax.typing import DTypeLike

from adaglimpse_nnx.config import AdaGlimpseConfig
from adaglimpse_nnx.positional_embedding import box_embedding, output_grid_embedding
from adaglimpse_nnx.tokenization import glimpse_patches, patch_boxes

Array = jax.Array

LAYER_NORM_EPSILON = 1e-6
LAYER_SCALE_INIT = 1e-4
TOKEN_INIT_STD = 0.02


def _linear(in_dim: int, out_dim: int, rngs: nnx.Rngs) -> nnx.Linear:
    """AdaGlimpse initializes every linear layer Xavier-uniform with zero bias."""
    return nnx.Linear(in_dim, out_dim, kernel_init=nnx.initializers.xavier_uniform(), rngs=rngs)


def _patch_embedding(*, patch_size_px: int, dim: int, rngs: nnx.Rngs) -> nnx.Linear:
    """Upstream embeds a patch with a convolution whose kernel is the patch: a linear map of its pixels.

    It redraws the kernel Xavier-uniform over that linear map and keeps
    PyTorch's default bias for a convolution, uniform within 1/sqrt(fan_in).
    """
    fan_in = 3 * patch_size_px**2
    bound = fan_in**-0.5

    def bias_init(key: Array, shape: tuple[int, ...], dtype: DTypeLike = jnp.float32) -> Array:
        return jax.random.uniform(key, shape, dtype, -bound, bound)

    return nnx.Linear(fan_in, dim, kernel_init=nnx.initializers.xavier_uniform(), bias_init=bias_init, rngs=rngs)


def _layer_norm(dim: int, rngs: nnx.Rngs) -> nnx.LayerNorm:
    return nnx.LayerNorm(dim, epsilon=LAYER_NORM_EPSILON, use_fast_variance=False, rngs=rngs)


class Attention(nnx.Module):
    """Multi-head self-attention with a biased query-key-value projection."""

    def __init__(self, dim: int, *, num_heads: int, rngs: nnx.Rngs) -> None:
        self.num_heads = num_heads
        self.qkv = _linear(dim, 3 * dim, rngs)
        self.proj = _linear(dim, dim, rngs)

    def __call__(self, x: Array) -> Array:
        batch, length, dim = x.shape
        head_dim = dim // self.num_heads
        qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, head_dim).transpose(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        weights = jax.nn.softmax(query @ jnp.swapaxes(key, -1, -2) * head_dim**-0.5, axis=-1)
        return self.proj(jnp.swapaxes(weights @ value, 1, 2).reshape(batch, length, dim))


class Mlp(nnx.Module):
    def __init__(self, dim: int, *, hidden_dim: int, rngs: nnx.Rngs) -> None:
        self.fc1 = _linear(dim, hidden_dim, rngs)
        self.fc2 = _linear(hidden_dim, dim, rngs)

    def __call__(self, x: Array) -> Array:
        return self.fc2(jax.nn.gelu(self.fc1(x), approximate=False))


class LayerScaleBlock(nnx.Module):
    """A pre-norm transformer block whose two residual branches each carry a learned per-channel scale."""

    def __init__(self, dim: int, *, num_heads: int, mlp_ratio: int, rngs: nnx.Rngs) -> None:
        self.norm1 = _layer_norm(dim, rngs)
        self.attn = Attention(dim, num_heads=num_heads, rngs=rngs)
        self.norm2 = _layer_norm(dim, rngs)
        self.mlp = Mlp(dim, hidden_dim=mlp_ratio * dim, rngs=rngs)
        self.gamma_1 = nnx.Param(jnp.full((dim,), LAYER_SCALE_INIT))
        self.gamma_2 = nnx.Param(jnp.full((dim,), LAYER_SCALE_INIT))

    def __call__(self, x: Array) -> Array:
        x = x + self.gamma_1[...] * self.attn(self.norm1(x))
        return x + self.gamma_2[...] * self.mlp(self.norm2(x))


class GlimpseSet(NamedTuple):
    """Every patch of a rollout so far, with the scene box each covers: AdaGlimpse's whole state."""

    patches: Array  # [B, L, P, P, 3]
    boxes: Array  # [B, L, 4] (top, left, bottom, right) in scene pixels, float32


class AdaGlimpseOutput(NamedTuple):
    state: GlimpseSet
    latent: Array  # [B, 1 + L, embed_dim]: the encoding of every patch in state, the class token first


class AdaGlimpseForPretraining(nnx.Module):
    """AdaGlimpse with its two output layers read as predictions of a teacher's features.

    `decoder_pred`, upstream the pixel reconstruction of each grid cell,
    predicts the teacher's patch features; `head`, upstream the classifier,
    predicts the teacher's CLS token.
    """

    def __init__(self, config: AdaGlimpseConfig, *, rngs: nnx.Rngs) -> None:
        self.config = config
        dim, decoder_dim = config.embed_dim, config.decoder_embed_dim
        self.patch_embed = _patch_embedding(patch_size_px=config.patch_size_px, dim=dim, rngs=rngs)
        self.cls_token = nnx.Param(TOKEN_INIT_STD * jax.random.normal(rngs.params(), (1, 1, dim)))
        self.blocks = nnx.List(
            [LayerScaleBlock(dim, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, rngs=rngs)
             for _ in range(config.depth)]
        )
        self.norm = _layer_norm(dim, rngs)
        self.decoder_embed = _linear(dim, decoder_dim, rngs)
        self.mask_token = nnx.Param(TOKEN_INIT_STD * jax.random.normal(rngs.params(), (1, 1, decoder_dim)))
        self.decoder_blocks = nnx.List(
            [LayerScaleBlock(decoder_dim, num_heads=config.decoder_num_heads, mlp_ratio=config.mlp_ratio, rngs=rngs)
             for _ in range(config.decoder_depth)]
        )
        self.decoder_norm = _layer_norm(decoder_dim, rngs)
        self.decoder_pred = _linear(decoder_dim, config.teacher_dim, rngs)
        self.head = _linear(dim, config.teacher_dim, rngs)

    def init_state(self, *, batch_size: int) -> GlimpseSet:
        """The empty set: AdaGlimpse starts a rollout with nothing learned."""
        assert batch_size > 0, batch_size
        patch = self.config.patch_size_px
        return GlimpseSet(
            patches=jnp.zeros((batch_size, 0, patch, patch, 3), jnp.float32),
            boxes=jnp.zeros((batch_size, 0, 4), jnp.float32),
        )

    def __call__(self, *, glimpse: Array, state: GlimpseSet, viewpoint: Viewpoint) -> AdaGlimpseOutput:
        """Add a [B, G, G, 3] glimpse, sampled at viewpoint, to the set, and encode the whole set."""
        config = self.config
        state = GlimpseSet(
            patches=jnp.concatenate(
                (state.patches, glimpse_patches(
                    glimpse, tokens_per_side=config.glimpse_tokens_per_side, patch_size_px=config.patch_size_px,
                )),
                axis=1,
            ),
            boxes=jnp.concatenate(
                (state.boxes, patch_boxes(
                    viewpoint, scene_size_px=config.scene_size_px, tokens_per_side=config.glimpse_tokens_per_side,
                )),
                axis=1,
            ),
        )
        return AdaGlimpseOutput(state=state, latent=self.encode(state.patches, state.boxes))

    def encode(self, patches: Array, boxes: Array) -> Array:
        """Encode [B, L, P, P, 3] patches covering [B, L, 4] scene boxes as [B, 1 + L, embed_dim].

        The class token comes first and carries no positional embedding.
        """
        batch, length = patches.shape[:2]
        x = self.patch_embed(patches.reshape(batch, length, -1))
        x = x + box_embedding(boxes, dim=self.config.embed_dim)
        cls = jnp.broadcast_to(self.cls_token[...], (batch, 1, self.config.embed_dim))
        x = jnp.concatenate((cls, x), axis=1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def decode(self, latent: Array) -> Array:
        """Decode a [B, 1 + L, embed_dim] encoding into [B, grid², decoder_embed_dim] features, a row-major grid.

        The decoder attends over the encoded tokens and a mask token per grid
        cell, which alone carries the cell's positional embedding, and keeps
        the cells. These are the features the architecture control's ADE20K
        probe reads.
        """
        batch = latent.shape[0]
        grid = output_grid_embedding(dim=self.config.decoder_embed_dim, grid_size=self.config.output_grid_size)
        cells = self.mask_token[...] + jnp.asarray(grid)[None]
        x = jnp.concatenate((self.decoder_embed(latent), jnp.broadcast_to(cells, (batch, *grid.shape))), axis=1)
        for block in self.decoder_blocks:
            x = block(x)
        return self.decoder_norm(x)[:, -grid.shape[0] :]

    def predict_teacher_patches(self, latent: Array) -> Array:
        """[B, grid², teacher_dim]: the teacher's patch features of the whole scene, from the decoded grid."""
        return self.decoder_pred(self.decode(latent))

    def predict_teacher_cls(self, latent: Array) -> Array:
        """[B, teacher_dim]: the teacher's CLS token, from the encoded class token."""
        return self.head(latent[:, 0])
