"""The fine-tuning gradient of the released classifier in bfloat16 against float32."""

import functools

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from canvit_nnx.model import CanViTForImageClassification, fused_canvas_attention
from canvit_nnx.model.config import ComputeDtype
from canvit_nnx.releases import DINOV3_VITB16_IN1K_PROBE, PRETRAINED, RELEASED_SCENE_SIZE_PX
from canvit_nnx.specialize.in1k.train import Recipe, draw_viewpoints, lpft_loss

BATCH_SIZE = 8


@pytest.mark.network
def test_the_bfloat16_gradient_through_pure_jax_canvas_attention_follows_float32(monkeypatch: pytest.MonkeyPatch) -> None:
    """The recipe's loss on the released CanViT-B, with Canvas Attention in JAX rather than the fused kernels.

    Its first Canvas Attention Write attends with keys whose norms reach 2e4, which makes its softmax backward
    nearly cancel; on TPU this gradient came out over a hundred times too large while the loss stayed right.
    """
    monkeypatch.setattr(fused_canvas_attention, "use_fused_kernels", lambda _: False)
    recipe = Recipe()
    scene_key, label_key = jax.random.split(jax.random.key(0))
    scenes = jax.random.normal(scene_key, (BATCH_SIZE, RELEASED_SCENE_SIZE_PX, RELEASED_SCENE_SIZE_PX, 3))
    labels = jax.random.randint(label_key, (BATCH_SIZE,), 0, 1000)
    viewpoints = draw_viewpoints(jax.random.key(recipe.seed), batch_size=BATCH_SIZE, num_glimpses=recipe.num_glimpses)
    loss = functools.partial(lpft_loss, label_smoothing=recipe.label_smoothing, canvas_grid_size=recipe.canvas_grid_size)

    def gradient(compute_dtype: ComputeDtype) -> jax.Array:
        classifier = CanViTForImageClassification.from_pretrained_with_probe(
            pretrained_repo=PRETRAINED[recipe.pretraining_dataset],
            probe_repo=DINOV3_VITB16_IN1K_PROBE,
            compute_dtype=compute_dtype,
        )
        with jax.default_matmul_precision("highest" if compute_dtype == "float32" else None):
            grads = nnx.jit(nnx.grad(lambda c: loss(c, scenes, labels, viewpoints)[0]))(classifier)
        return jnp.concatenate([jnp.ravel(leaf) for leaf in jax.tree.leaves(grads)]).astype(jnp.float32)

    expected, actual = gradient("float32"), gradient("bfloat16")
    cosine = float(expected @ actual / (jnp.linalg.norm(expected) * jnp.linalg.norm(actual)))
    norm_ratio = float(jnp.linalg.norm(actual) / jnp.linalg.norm(expected))
    # Measured: cosine 0.9996 and norm ratio 0.996 on CPU, 0.9998 and 0.994 on TPU; -0.07 and 134 on TPU before
    # the score cotangent was computed apart.
    assert cosine > 0.99 and abs(norm_ratio - 1) < 0.1, (cosine, norm_ratio)
