"""AdaGlimpse's tokenization of a glimpse: square crops resized to patches, each with the scene box it covers."""

import jax
import jax.numpy as jnp
from canvit_nnx.viewpoint import Viewpoint, crop_box_px

Array = jax.Array


def glimpse_patches(glimpse: Array, *, tokens_per_side: int, patch_size_px: int) -> Array:
    """Cut a [B, G, G, C] glimpse into tokens_per_side² square crops, row-major, each resized to a patch.

    Returns [B, tokens_per_side², patch_size_px, patch_size_px, C]. As
    AdaGlimpse's sampler does (`_crop_and_resize` upstream), a crop of a
    different size is resized bilinearly without antialiasing, and a crop of
    the patch size passes through unchanged.
    """
    batch, height, width, channels = glimpse.shape
    assert height == width and height % tokens_per_side == 0, (glimpse.shape, tokens_per_side)
    crop = height // tokens_per_side
    crops = glimpse.reshape(batch, tokens_per_side, crop, tokens_per_side, crop, channels)
    crops = crops.transpose(0, 1, 3, 2, 4, 5).reshape(batch, tokens_per_side**2, crop, crop, channels)
    if crop == patch_size_px:
        return crops
    shape = (batch, tokens_per_side**2, patch_size_px, patch_size_px, channels)
    return jax.image.resize(crops, shape, method="bilinear", antialias=False)


def patch_boxes(viewpoint: Viewpoint, *, scene_size_px: int, tokens_per_side: int) -> Array:
    """The scene box (top, left, bottom, right), in pixels, of each crop `glimpse_patches` cuts at viewpoint.

    Returns [B, tokens_per_side², 4] float32, in the crops' order. Upstream's
    sampler rounds a crop down to whole pixels and gives its last row and
    column of crops the remainder; the glimpse here is sampled continuously
    (`canvit_nnx.sample_at_viewpoint`), so its crop box is cut evenly and each
    box is exactly the scene area its patch shows.
    """
    box = crop_box_px(viewpoint, image_size_px=scene_size_px)
    top_left, step = box[:, None, :2], (box[:, None, 2:] - box[:, None, :2]) / tokens_per_side
    index = jnp.arange(tokens_per_side, dtype=jnp.float32)
    rows, columns = jnp.meshgrid(index, index, indexing="ij")
    starts = top_left + jnp.stack((rows.ravel(), columns.ravel()), axis=-1) * step
    return jnp.concatenate((starts, starts + step), axis=-1)
