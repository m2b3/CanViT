"""AdaGlimpse's sine-cosine positional embeddings: of a patch by its scene box, and of the decoder's output grid."""

import functools

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array

BOX_UNIT_PX = 16.0
"""AdaGlimpse divides box coordinates by 16 pixels before embedding them, whatever the patch size."""


def box_embedding(boxes: Array, *, dim: int) -> Array:
    """Embed [..., 4] scene boxes (top, left, bottom, right), in pixels, as [..., dim] float32.

    Each coordinate takes a quarter of the width, in that order, half sines
    and half cosines (`get_2dplus_sincos_pos_embed_coords` upstream).
    """
    assert boxes.shape[-1] == 4 and dim % 8 == 0, (boxes.shape, dim)
    half_quarter = dim // 8
    frequencies = 1.0 / 10000.0 ** (jnp.arange(half_quarter, dtype=jnp.float32) / half_quarter)
    angles = (boxes.astype(jnp.float32) / BOX_UNIT_PX)[..., None] * frequencies
    return jnp.concatenate((jnp.sin(angles), jnp.cos(angles)), axis=-1).reshape(*boxes.shape[:-1], dim)


@functools.cache
def output_grid_embedding(*, dim: int, grid_size: int) -> np.ndarray:
    """Embed the decoder's grid_size × grid_size output cells, row-major, as a read-only [grid_size², dim] float32.

    Computed in NumPy float32 as upstream computes it (`get_2d_sincos_pos_embed`);
    the first half of the width embeds the column and the second the row.
    """
    assert dim % 4 == 0, dim
    index = np.arange(grid_size, dtype=np.float32)
    columns, rows = np.meshgrid(index, index)
    embedding = np.concatenate((_sincos(columns.ravel(), dim // 2), _sincos(rows.ravel(), dim // 2)), axis=1)
    embedding.flags.writeable = False
    return embedding


def _sincos(positions: np.ndarray, dim: int) -> np.ndarray:
    frequencies = np.arange(dim // 2, dtype=np.float32)
    frequencies /= dim / 2.0
    frequencies = 1.0 / 10000**frequencies
    angles = np.einsum("m,d->md", positions, frequencies)
    return np.concatenate((np.sin(angles), np.cos(angles)), axis=1)
