import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from canvit_nnx.viewpoint import Viewpoint, sample_at_viewpoint

from adaglimpse_nnx import AdaGlimpseConfig, AdaGlimpseForPretraining, GlimpseSet
from adaglimpse_nnx.tokenization import glimpse_patches, patch_boxes

TINY = AdaGlimpseConfig(
    scene_size_px=128, glimpse_tokens_per_side=2, teacher_dim=8, embed_dim=32, depth=2, num_heads=2,
    decoder_embed_dim=16, decoder_depth=2, decoder_num_heads=2,
)


@pytest.mark.parametrize("tokens_per_side", [2, 8])
@pytest.mark.parametrize("scale", [1.0, 0.73, 0.31])
def test_each_box_is_the_scene_area_its_patch_shows(tokens_per_side: int, scale: float) -> None:
    """On a scene whose pixels hold their own row and column, a patch's mean is the centre of its box.

    The scales leave crops that are not whole pixels, where upstream's integer
    boxes would not describe the pixels of a glimpse sampled continuously.
    """
    size = 128
    index = jnp.arange(size, dtype=jnp.float32)
    rows, columns = jnp.meshgrid(index, index, indexing="ij")
    scene = jnp.stack((rows, columns, jnp.zeros((size, size))), axis=-1)[None]
    viewpoint = Viewpoint(centers=jnp.asarray([[0.2 * (1 - scale), -0.1 * (1 - scale)]], jnp.float32),
                          scales=jnp.asarray([scale], jnp.float32))
    glimpse = sample_at_viewpoint(spatial=scene, viewpoint=viewpoint, glimpse_size_px=128)
    patches = glimpse_patches(glimpse, tokens_per_side=tokens_per_side, patch_size_px=16)
    boxes = patch_boxes(viewpoint, scene_size_px=size, tokens_per_side=tokens_per_side)
    # Pixel i covers [i, i + 1), so a box's centre in pixel indices is its midpoint less a half.
    centres = (boxes[0, :, :2] + boxes[0, :, 2:]) / 2 - 0.5
    np.testing.assert_allclose(patches[0, ..., :2].mean(axis=(1, 2)), centres, atol=0.05)


def test_a_compiled_rollout_matches_the_eager_one() -> None:
    """Three glimpses, one compiled step each: the set grows, and compilation changes nothing."""
    model = AdaGlimpseForPretraining(TINY, rngs=nnx.Rngs(0))
    rng = np.random.default_rng(0)
    scene = jnp.asarray(rng.normal(size=(2, TINY.scene_size_px, TINY.scene_size_px, 3)), jnp.float32)
    viewpoints = [Viewpoint.full_scene(batch_size=2), Viewpoint(
        centers=jnp.asarray(rng.uniform(-0.5, 0.5, (2, 2)), jnp.float32), scales=jnp.full((2,), 0.5, jnp.float32),
    ), Viewpoint(centers=jnp.zeros((2, 2), jnp.float32), scales=jnp.full((2,), 0.25, jnp.float32))]

    @nnx.jit
    def step(model: AdaGlimpseForPretraining, glimpse: jax.Array, state: GlimpseSet, viewpoint: Viewpoint):
        output = model(glimpse=glimpse, state=state, viewpoint=viewpoint)
        return output.state, model.predict_teacher_patches(output.latent), model.predict_teacher_cls(output.latent)

    eager_state = compiled_state = model.init_state(batch_size=2)
    for t, viewpoint in enumerate(viewpoints, start=1):
        glimpse = sample_at_viewpoint(spatial=scene, viewpoint=viewpoint, glimpse_size_px=32)
        output = model(glimpse=glimpse, state=eager_state, viewpoint=viewpoint)
        eager_state = output.state
        compiled_state, teacher_patches, teacher_cls = step(model, glimpse, compiled_state, viewpoint)
        assert eager_state.patches.shape[1] == eager_state.boxes.shape[1] == t * TINY.tokens_per_glimpse
        np.testing.assert_allclose(teacher_patches, model.predict_teacher_patches(output.latent), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(teacher_cls, model.predict_teacher_cls(output.latent), rtol=1e-5, atol=1e-6)
        assert teacher_patches.shape == (2, TINY.output_grid_size**2, TINY.teacher_dim)
