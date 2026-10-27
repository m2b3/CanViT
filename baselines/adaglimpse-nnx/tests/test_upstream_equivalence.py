"""This package against AdaGlimpse's own modules at the pinned commit, at the architecture control's configuration."""

import copy
import math
import types
from collections.abc import Callable
from typing import Any

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
import pytest
import reference
import torch
from canvit_nnx.hub import load_state
from canvit_nnx.viewpoint import Viewpoint, sample_at_viewpoint
from safetensors.numpy import save_file
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from adaglimpse_nnx import ARCHITECTURE_CONTROL, AdaGlimpseForPretraining, flops
from adaglimpse_nnx.model import LAYER_NORM_EPSILON
from adaglimpse_nnx.positional_embedding import box_embedding, output_grid_embedding
from adaglimpse_nnx.tokenization import glimpse_patches, patch_boxes

pytestmark = pytest.mark.network

CONFIG = ARCHITECTURE_CONTROL
# Measured near 1e-6 in float32 on CPU; a misplaced weight or operation moves outputs by orders of magnitude more.
TOLERANCE = 1e-5


def _relative_error(actual: object, expected: object) -> float:
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    return float(np.abs(actual - expected).max() / np.abs(expected).max())


def _parameters(model: AdaGlimpseForPretraining) -> dict[str, np.ndarray]:
    state = nnx.to_flat_state(nnx.state(model, nnx.Param))
    return {".".join(map(str, path)): np.asarray(variable[...])
            for path, variable in zip(state.paths, state.leaves, strict=True)}


def test_upstream_hyperparameters_without_a_shape(upstream_model: Any) -> None:
    """Loading upstream weights checks every width and depth; these are what no tensor shape records."""
    assert upstream_model.blocks[0].attn.num_heads == CONFIG.num_heads
    assert upstream_model.decoder_blocks[0].attn.num_heads == CONFIG.decoder_num_heads
    assert {m.eps for m in upstream_model.modules() if isinstance(m, nn.LayerNorm)} == {LAYER_NORM_EPSILON}
    frozen = {name for name, parameter in upstream_model.named_parameters() if not parameter.requires_grad}
    assert frozen == reference.FIXED_TABLES


def test_initialization_draws_from_upstreams_distributions(upstream_model: Any) -> None:
    """Every parameter starts as upstream's does: the same constant, or a draw of the same spread and range.

    Loading upstream weights overwrites every initial value, so the forward
    comparison cannot see an initialization defect; training from scratch
    inherits it. The spread of two draws differs by about 1/sqrt(n) of itself.
    """
    ours = _parameters(AdaGlimpseForPretraining(CONFIG, rngs=nnx.Rngs(0)))
    theirs = reference.package_tensors(upstream_model)
    assert ours.keys() == theirs.keys()
    for key, expected in theirs.items():
        actual = ours[key]
        if np.all(expected == expected.flat[0]):
            assert np.all(actual == expected.flat[0]), key
            continue
        assert abs(actual.std() / expected.std() - 1) < 4 / math.sqrt(expected.size), key
        if expected.size >= 10_000:
            assert abs(np.abs(actual).max() / np.abs(expected).max() - 1) < 0.05, key


@pytest.fixture(scope="module")
def models(upstream_model: Any, tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, AdaGlimpseForPretraining]:
    """Upstream's model with its biases, norms and layer scales redrawn, so that each shapes the output, and this
    package's model holding the same weights."""
    theirs = copy.deepcopy(upstream_model)
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for parameter in theirs.parameters():
            if parameter.ndim == 1:
                parameter.normal_(0.5, 0.25, generator=generator)
    path = tmp_path_factory.mktemp("weights") / "upstream.safetensors"
    save_file(reference.package_tensors(theirs), str(path))
    ours = AdaGlimpseForPretraining(CONFIG, rngs=nnx.Rngs(0))
    load_state(ours, path)
    return theirs, ours


@pytest.mark.parametrize("num_glimpses", [1, 3])
def test_forward_matches_upstream(models: tuple[Any, AdaGlimpseForPretraining], num_glimpses: int) -> None:
    """Encoding, decoding and both readouts of the same patches and boxes; a full-scene glimpse first."""
    theirs, ours = models
    rng = np.random.default_rng(num_glimpses)
    batch_size = 2
    viewpoints = [Viewpoint.full_scene(batch_size=batch_size)] + [
        Viewpoint(centers=jnp.asarray(rng.uniform(-0.6, 0.6, (batch_size, 2)), jnp.float32),
                  scales=jnp.asarray(rng.uniform(0.1, 0.5, batch_size), jnp.float32))
        for _ in range(num_glimpses - 1)
    ]
    boxes = jnp.concatenate([
        patch_boxes(viewpoint, scene_size_px=CONFIG.scene_size_px, tokens_per_side=CONFIG.glimpse_tokens_per_side)
        for viewpoint in viewpoints
    ], axis=1)
    patch = CONFIG.patch_size_px
    patches = rng.normal(size=(batch_size, boxes.shape[1], patch, patch, 3)).astype(np.float32)

    with torch.no_grad():
        their_latent, their_box_embedding = theirs.forward_encoder(
            torch.from_numpy(patches.transpose(0, 1, 4, 2, 3).copy()), coords=torch.from_numpy(np.array(boxes)),
        )
        their_teacher_patches = theirs.forward_decoder(their_latent)
        their_teacher_cls = theirs.forward_head(their_latent)
    latent = ours.encode(jnp.asarray(patches), boxes)

    assert _relative_error(box_embedding(boxes, dim=CONFIG.embed_dim), their_box_embedding) < TOLERANCE
    assert _relative_error(latent, their_latent) < TOLERANCE
    assert _relative_error(ours.predict_teacher_patches(latent), their_teacher_patches) < TOLERANCE
    assert _relative_error(ours.predict_teacher_cls(latent), their_teacher_cls) < TOLERANCE


def test_output_grid_embedding_is_upstreams(upstream_model: Any) -> None:
    """The decoder's fixed table, computed here rather than loaded, is upstream's bit for bit."""
    grid = output_grid_embedding(dim=CONFIG.decoder_embed_dim, grid_size=CONFIG.output_grid_size)
    assert np.array_equal(grid, upstream_model.decoder_pos_embed[0, 1:].numpy())


@pytest.mark.parametrize("tokens_per_side", [2, 4, 8])
def test_tokenization_matches_upstreams_sampler(upstream: types.SimpleNamespace, tokens_per_side: int) -> None:
    """A glimpse that shows its crop at the scene's resolution is cut into the patches, and the scene boxes, that
    AdaGlimpse's sampler cuts from the scene itself; and so is the full scene, by its boxes."""
    scene_size, glimpse_size, patch = CONFIG.scene_size_px, 128, CONFIG.patch_size_px
    sampler = upstream.sampler.InteractiveStatelessSampler(
        glimpse_grid_size=tokens_per_side, max_glimpses=1, native_patch_size=(patch, patch),
    )
    scene = np.random.default_rng(tokens_per_side).random((1, scene_size, scene_size, 3), dtype=np.float32)

    def theirs(action: list[float]) -> tuple[np.ndarray, np.ndarray]:
        """Upstream's crops for an action (top and left as fractions of the free range, then side), then patches."""
        crops = sampler._calculate_crops(torch.tensor([action], dtype=torch.float64), (1, 3, scene_size, scene_size))
        patches, boxes, _ = sampler._crop_and_resize(
            crops, torch.from_numpy(scene.transpose(0, 3, 1, 2).copy()), torch.zeros(1),
        )
        return patches.numpy().transpose(0, 1, 3, 4, 2), boxes.numpy()

    # A glimpse-sized crop at whole pixels: its glimpse samples the scene's own pixels.
    top, left, min_side, free = 96, 192, patch * tokens_per_side, scene_size - glimpse_size
    their_patches, their_boxes = theirs([top / free, left / free, (glimpse_size - min_side) / (scene_size - min_side)])
    crop = glimpse_size // tokens_per_side
    assert their_boxes[0, 0].tolist() == [top, left, top + crop, left + crop], "upstream took another crop"
    centre = [(corner + glimpse_size / 2) / (scene_size / 2) - 1 for corner in (top, left)]
    viewpoint = Viewpoint(centers=jnp.asarray([centre], jnp.float32),
                          scales=jnp.asarray([glimpse_size / scene_size], jnp.float32))
    glimpse = sample_at_viewpoint(spatial=jnp.asarray(scene), viewpoint=viewpoint, glimpse_size_px=glimpse_size)
    patches = glimpse_patches(glimpse, tokens_per_side=tokens_per_side, patch_size_px=patch)
    assert _relative_error(patches, their_patches) < TOLERANCE
    boxes = patch_boxes(viewpoint, scene_size_px=scene_size, tokens_per_side=tokens_per_side)
    assert np.array_equal(boxes, their_boxes)

    _, their_full_scene_boxes = theirs([0.0, 0.0, 1.0])
    full_scene_boxes = patch_boxes(Viewpoint.full_scene(batch_size=1), scene_size_px=scene_size,
                                   tokens_per_side=tokens_per_side)
    assert np.array_equal(full_scene_boxes, their_full_scene_boxes)


def _counted_flops(forward: Callable[[], object]) -> int:
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        forward()
    return counter.get_total_flops()


@pytest.mark.parametrize("num_glimpses", [1, 2, 3])
def test_flops_equal_a_count_of_upstreams_forward(upstream_model: Any, num_glimpses: int) -> None:
    """torch.utils.flop_counter over upstream's three calls per glimpse, against the analytic counts."""
    num_patches = num_glimpses * CONFIG.tokens_per_glimpse
    patches = torch.randn(1, num_patches, 3, CONFIG.patch_size_px, CONFIG.patch_size_px)
    boxes = torch.rand(1, num_patches, 4) * CONFIG.scene_size_px
    with torch.no_grad():
        latent, _ = upstream_model.forward_encoder(patches, coords=boxes)

    patch_readout = flops.linear_flops(
        num_tokens=CONFIG.output_grid_size**2, in_dim=CONFIG.decoder_embed_dim, out_dim=CONFIG.teacher_dim,
    )
    cls_readout = flops.linear_flops(num_tokens=1, in_dim=CONFIG.embed_dim, out_dim=CONFIG.teacher_dim)
    assert _counted_flops(lambda: upstream_model.forward_encoder(patches, coords=boxes)) == flops.encoder_flops(
        CONFIG, num_glimpses=num_glimpses
    )
    assert _counted_flops(lambda: upstream_model.forward_decoder(latent)) == flops.decoder_flops(
        CONFIG, num_glimpses=num_glimpses
    ) + patch_readout
    assert _counted_flops(lambda: upstream_model.forward_head(latent)) == cls_readout
