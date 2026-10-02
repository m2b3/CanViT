"""Scene coordinates, viewpoints, and native NHWC bilinear sampling."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array


class Viewpoint(NamedTuple):
    centers: Array
    scales: Array

    @staticmethod
    def full_scene(*, batch_size: int) -> "Viewpoint":
        assert batch_size > 0, batch_size
        return Viewpoint(
            centers=jnp.zeros((batch_size, 2), dtype=jnp.float32),
            scales=jnp.ones((batch_size,), dtype=jnp.float32),
        )


def _validate_viewpoint(viewpoint: Viewpoint) -> None:
    assert viewpoint.centers.dtype == viewpoint.scales.dtype == jnp.float32, (
        viewpoint.centers.dtype,
        viewpoint.scales.dtype,
    )
    assert viewpoint.centers.ndim == 2 and viewpoint.scales.ndim == 1, (
        viewpoint.centers.shape,
        viewpoint.scales.shape,
    )
    assert viewpoint.centers.shape == (viewpoint.scales.shape[0], 2), (
        viewpoint.centers.shape,
        viewpoint.scales.shape,
    )


def grid_coords(*, size: int) -> Array:
    """Cell centers of a square grid over the scene frame, in row/column order."""
    assert size > 0, size
    centers = (jnp.arange(size, dtype=jnp.float32) + 0.5) / size * 2.0 - 1.0
    rows, cols = jnp.meshgrid(centers, centers, indexing="ij")
    return jnp.stack((rows, cols), axis=-1)


def viewpoint_grid_coords(viewpoint: Viewpoint, *, size: int) -> Array:
    """Scene coordinates of the square sample grid for each viewpoint."""
    _validate_viewpoint(viewpoint)
    offsets = grid_coords(size=size)
    return viewpoint.centers[:, None, None, :] + viewpoint.scales[:, None, None, None] * offsets[None, :, :, :]


def crop_box_px(viewpoint: Viewpoint, *, image_size_px: int) -> Array:
    assert image_size_px > 0, image_size_px
    _validate_viewpoint(viewpoint)
    half_side = viewpoint.scales[:, None]
    corners = jnp.concatenate(
        (viewpoint.centers - half_side, viewpoint.centers + half_side),
        axis=1,
    )
    return (corners + 1.0) * (image_size_px / 2.0)


def sample_at_viewpoint(*, spatial: Array, viewpoint: Viewpoint, glimpse_size_px: int) -> Array:
    """Sample zero-padded bilinear crops from an NHWC image or feature map.

    Pixel coordinates use the ``align_corners=False`` mapping used by
    ``torch.nn.functional.grid_sample``. Computation is float32 and the input
    dtype is restored on return.
    """
    assert spatial.ndim == 4, spatial.shape
    batch_size, height, width, _ = spatial.shape
    assert min(batch_size, height, width) > 0, spatial.shape
    assert glimpse_size_px > 0, glimpse_size_px
    _validate_viewpoint(viewpoint)
    assert batch_size == viewpoint.scales.shape[0], (spatial.shape, viewpoint.scales.shape)

    scene_positions = viewpoint_grid_coords(viewpoint, size=glimpse_size_px)
    image_size = jnp.asarray((height, width), dtype=jnp.float32)
    pixels = (scene_positions + 1.0) * image_size / 2.0 - 0.5
    lower = jnp.floor(pixels).astype(jnp.int32)
    fraction = pixels - lower.astype(jnp.float32)

    row, col = lower[..., 0], lower[..., 1]
    row_fraction = fraction[..., 0, None]
    col_fraction = fraction[..., 1, None]
    spatial_float = spatial.astype(jnp.float32)
    batch_indices = jnp.arange(batch_size, dtype=jnp.int32)[:, None, None]

    def gather(row_index: Array, col_index: Array) -> Array:
        valid = (row_index >= 0) & (row_index < height) & (col_index >= 0) & (col_index < width)
        values = spatial_float[
            batch_indices,
            jnp.clip(row_index, 0, height - 1),
            jnp.clip(col_index, 0, width - 1),
        ]
        return jnp.where(valid[..., None], values, 0.0)

    sampled = (
        gather(row, col) * (1.0 - row_fraction) * (1.0 - col_fraction)
        + gather(row, col + 1) * (1.0 - row_fraction) * col_fraction
        + gather(row + 1, col) * row_fraction * (1.0 - col_fraction)
        + gather(row + 1, col + 1) * row_fraction * col_fraction
    )
    return sampled.astype(spatial.dtype)
