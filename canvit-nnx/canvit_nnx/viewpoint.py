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


def _interpolation_weights(positions: Array, *, size: int) -> Array:
    """[..., size] bilinear weights of scene positions [...] over the pixels of an axis of that size.

    Pixel coordinates use the ``align_corners=False`` mapping of
    ``torch.nn.functional.grid_sample``; a sample position draws on at most two
    pixels, and pixels outside the axis are absent, which zero-pads.
    """
    pixels = (positions + 1.0) * (size / 2.0) - 0.5
    return jnp.maximum(0.0, 1.0 - jnp.abs(pixels[..., None] - jnp.arange(size, dtype=jnp.float32)))


def sample_at_viewpoint(*, spatial: Array, viewpoint: Viewpoint, glimpse_size_px: int) -> Array:
    """Sample zero-padded bilinear crops from an NHWC image or feature map.

    Pixel coordinates use the ``align_corners=False`` mapping used by
    ``torch.nn.functional.grid_sample``. The interpolation is separable, a
    product with row weights then with column weights, in the input dtype
    with float32 accumulation: matrix products instead of the per-pixel
    gathers that TPUs execute slowly. Returns the input dtype.
    """
    assert spatial.ndim == 4, spatial.shape
    batch_size, height, width, _ = spatial.shape
    assert min(batch_size, height, width) > 0, spatial.shape
    assert glimpse_size_px > 0, glimpse_size_px
    _validate_viewpoint(viewpoint)
    assert batch_size == viewpoint.scales.shape[0], (spatial.shape, viewpoint.scales.shape)

    offsets = grid_coords(size=glimpse_size_px)[:, 0, 0]
    rows = viewpoint.centers[:, 0, None] + viewpoint.scales[:, None] * offsets
    cols = viewpoint.centers[:, 1, None] + viewpoint.scales[:, None] * offsets
    sampled = jnp.einsum(
        "byh,bhwc,bxw->byxc",
        _interpolation_weights(rows, size=height).astype(spatial.dtype),
        spatial,
        _interpolation_weights(cols, size=width).astype(spatial.dtype),
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )
    return sampled.astype(spatial.dtype)
