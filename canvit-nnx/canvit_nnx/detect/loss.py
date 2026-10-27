"""CenterNet's training objective ("Objects as Points", arXiv:1904.07850).

A penalty-reduced focal loss on the heatmap, normalized by the number of
object centers, plus L1 losses on size and offset at the center cells.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from canvit_nnx.detect.model import CenterNetMaps
from canvit_nnx.detect.targets import Targets

Array = jax.Array

FOCAL_ALPHA = 2.0
FOCAL_BETA = 4.0
"""Down-weights negatives near a center by (1 - target)^FOCAL_BETA."""
OFFSET_WEIGHT = 1.0
CENTERNET_SIZE_WEIGHT = 0.1
CENTERNET_OUTPUT_STRIDE = 4
"""CenterNet's size weight is for its output stride in pixels; sizes are regressed in cells."""


def size_weight(*, output_stride: float) -> float:
    """CenterNet's size-loss weight at another output stride: a size in cells, and its error, scale as 1 / stride."""
    return CENTERNET_SIZE_WEIGHT * output_stride / CENTERNET_OUTPUT_STRIDE


class LossTerms(NamedTuple):
    heatmap: Array
    size: Array
    offset: Array


def focal_loss(logits: Array, heatmap: Array) -> Array:
    """Penalty-reduced focal loss of heatmap logits, summed and divided by the number of centers (at least 1).

    The centers are the cells where the target is exactly 1; the loss uses
    log-sigmoids of the logits, which stay finite for confident cells.
    """
    is_center = heatmap == 1.0
    probability = jax.nn.sigmoid(logits)
    center_loss = -((1.0 - probability) ** FOCAL_ALPHA) * jax.nn.log_sigmoid(logits)
    other_loss = -((1.0 - heatmap) ** FOCAL_BETA) * probability**FOCAL_ALPHA * jax.nn.log_sigmoid(-logits)
    return jnp.sum(jnp.where(is_center, center_loss, other_loss)) / jnp.maximum(jnp.sum(is_center), 1.0)


def masked_l1(prediction: Array, target: Array, mask: Array) -> Array:
    """Mean absolute error over the masked cells' [B, G, G, C] channels, as CenterNet's RegL1Loss."""
    error = jnp.sum(jnp.abs(prediction - target), axis=-1)
    return jnp.sum(error * mask) / jnp.maximum(jnp.sum(mask) * prediction.shape[-1], 1.0)


def detection_loss(maps: CenterNetMaps, targets: Targets[Array], *, size_weight: float) -> tuple[Array, LossTerms]:
    """The weighted sum of CenterNet's three terms over a batch, and the terms."""
    terms = LossTerms(
        heatmap=focal_loss(maps.heatmap, targets.heatmap),
        size=masked_l1(maps.size, targets.size, targets.mask),
        offset=masked_l1(maps.offset, targets.offset, targets.mask),
    )
    return terms.heatmap + size_weight * terms.size + OFFSET_WEIGHT * terms.offset, terms
