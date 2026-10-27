"""CenterNet decoding: heatmap peaks to a fixed number of boxes per scene, without non-maximum suppression."""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from canvit_nnx.detect.model import CenterNetMaps

Array = jax.Array

MAX_DETECTIONS = 100
"""Detections per scene; pycocotools scores at most 100 (maxDets)."""


class Detections[A: (np.ndarray, jax.Array)](NamedTuple):
    """K detections per scene, on the device or the host; arrays may have further leading axes."""

    boxes: A  # float32 [B, K, 4] normalized (ymin, xmin, ymax, xmax)
    scores: A  # float32 [B, K] heatmap probabilities, in decreasing order
    labels: A  # int32 [B, K] class indices


def _peaks(probabilities: Array) -> Array:
    """The probabilities that are maximal in their 3 × 3 neighborhood of the same class; zero elsewhere."""
    neighborhood_max = jax.lax.reduce_window(
        probabilities, -jnp.inf, jax.lax.max, (1, 3, 3, 1), (1, 1, 1, 1), ((0, 0), (1, 1), (1, 1), (0, 0))
    )
    return jnp.where(neighborhood_max == probabilities, probabilities, 0.0)


# Jitted: on TPU v6e (jax 0.11), an eager lax.top_k on a device-committed array returned zero indices.
@functools.partial(jax.jit, static_argnames=("max_detections",))
def decode(maps: CenterNetMaps, *, max_detections: int = MAX_DETECTIONS) -> Detections[Array]:
    """The max_detections highest heatmap peaks of each scene, each with the box its cell predicts."""
    batch_size, grid_size, _, num_classes = maps.heatmap.shape
    scores, indices = jax.lax.top_k(_peaks(jax.nn.sigmoid(maps.heatmap)).reshape(batch_size, -1), max_detections)
    cells, labels = indices // num_classes, indices % num_classes

    def at_cells(values: Array) -> Array:
        return jnp.take_along_axis(values.reshape(batch_size, grid_size * grid_size, 2), cells[..., None], axis=1)

    offset, size = at_cells(maps.offset), at_cells(maps.size)
    center_row = (cells // grid_size).astype(offset.dtype) + offset[..., 0]
    center_col = (cells % grid_size).astype(offset.dtype) + offset[..., 1]
    half_height, half_width = size[..., 0] / 2, size[..., 1] / 2
    boxes = jnp.stack(
        (center_row - half_height, center_col - half_width, center_row + half_height, center_col + half_width), axis=-1
    )
    return Detections(boxes=boxes / grid_size, scores=scores, labels=labels)
