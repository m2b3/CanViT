"""Viewing policies: where CanViT looks next (paper, `sec:experiments` and `supp:policy-definitions`)."""

import jax
import jax.numpy as jnp
import numpy as np

from canvit_nnx.viewpoint import Viewpoint

MIN_SCALE = 0.05
"""Smallest viewpoint scale of pretraining: a glimpse covering 0.25% of the scene."""


def random_viewpoint(key: jax.Array, *, batch_size: int, min_scale: float = MIN_SCALE) -> Viewpoint:
    """A random viewpoint per scene, as sampled in pretraining (paper, `subsec:policy-agnosticism`).

    Scales have density p(s) ∝ 1 - s on [min_scale, 1]: the half side h of the
    box of centers whose crops fit in the scene is drawn with p(h) ∝ h, then
    s = 1 - h and the center is uniform in that box.
    """
    scale_key, center_key = jax.random.split(key)
    center_box_half_side = jnp.sqrt(jax.random.uniform(scale_key, (batch_size,))) * (1.0 - min_scale)
    centers = jax.random.uniform(center_key, (batch_size, 2), minval=-1.0, maxval=1.0) * center_box_half_side[:, None]
    return Viewpoint(centers=centers, scales=1.0 - center_box_half_side)


def full_then_random(
    key: jax.Array, *, batch_size: int, num_glimpses: int, min_scale: float = MIN_SCALE
) -> tuple[Viewpoint, ...]:
    """F-IID: the full scene, then random viewpoints; one of the two pretraining rollouts."""
    assert num_glimpses >= 1, num_glimpses
    keys = jax.random.split(key, num_glimpses - 1)
    return (
        Viewpoint.full_scene(batch_size=batch_size),
        *(random_viewpoint(glimpse_key, batch_size=batch_size, min_scale=min_scale) for glimpse_key in keys),
    )


def all_random(
    key: jax.Array, *, batch_size: int, num_glimpses: int, min_scale: float = MIN_SCALE
) -> tuple[Viewpoint, ...]:
    """R-IID: random viewpoints from the first glimpse on; the other pretraining rollout."""
    assert num_glimpses >= 1, num_glimpses
    keys = jax.random.split(key, num_glimpses)
    return tuple(random_viewpoint(glimpse_key, batch_size=batch_size, min_scale=min_scale) for glimpse_key in keys)


def quadtree_level(level: int) -> np.ndarray:
    """[4**level, 3] (row, col, scale) of the level's tiles: a 2**level-per-side grid of crops, row-major."""
    n = 2**level
    centers = (2 * np.arange(n) + 1) / n - 1.0
    rows, cols = np.meshgrid(centers, centers, indexing="ij")
    return np.stack((rows.ravel(), cols.ravel(), np.full(n * n, 1.0 / n)), axis=1).astype(np.float32)


def coarse_to_fine(key: jax.Array, *, batch_size: int, num_glimpses: int) -> tuple[Viewpoint, ...]:
    """C2F: the full scene, then its quadrants, then sixteenths, each level's tiles in a random order per scene."""
    assert num_glimpses >= 1, num_glimpses
    viewpoints: list[Viewpoint] = []
    level = 0
    while len(viewpoints) < num_glimpses:
        tiles = jnp.asarray(quadtree_level(level))
        key, level_key = jax.random.split(key)
        chosen = tiles[jnp.argsort(jax.random.uniform(level_key, (batch_size, len(tiles))), axis=-1)]
        for index in range(min(len(tiles), num_glimpses - len(viewpoints))):
            viewpoints.append(Viewpoint(centers=chosen[:, index, :2], scales=chosen[:, index, 2]))
        level += 1
    return tuple(viewpoints)


__all__ = ["MIN_SCALE", "all_random", "coarse_to_fine", "full_then_random", "quadtree_level", "random_viewpoint"]
