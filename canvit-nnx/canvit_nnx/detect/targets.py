"""CenterNet training targets ("Objects as Points", arXiv:1904.07850), rendered from boxes on the host.

Boxes are normalized (ymin, xmin, ymax, xmax) in [0, 1]. The targets live on the
detector's output grid, one cell per canvas patch.
"""

from typing import NamedTuple

import jax
import numpy as np

MIN_OVERLAP = 0.7
"""IoU that a box displaced by the Gaussian radius keeps with the true box, as in CenterNet."""


class Targets[A: (np.ndarray, jax.Array)](NamedTuple):
    """Dense targets over a G × G grid; a batch stacks them along a leading axis."""

    heatmap: A  # float32 [G, G, num_classes]: a Gaussian of peak 1 at each object's center cell
    size: A  # float32 [G, G, 2]: (height, width) of the box centered in the cell, in cells
    offset: A  # float32 [G, G, 2]: the center minus the corner of its cell, in cells
    mask: A  # float32 [G, G]: 1 at the cells holding a center, which carry size and offset


def gaussian_radius(height: float, width: float) -> float:
    """CenterNet's Gaussian radius, in cells, for a box of that size; at least 1.

    The smallest displacement over three cases (a box of the same size shifted,
    enclosed or enclosing) that keeps MIN_OVERLAP, solved with the roots of
    CenterNet's src/lib/utils/image.py::gaussian_radius. Radius 0 would leave the
    focal loss no penalty reduction around a center, hence the floor.
    """
    b1 = height + width
    c1 = width * height * (1 - MIN_OVERLAP) / (1 + MIN_OVERLAP)
    r1 = (b1 + np.sqrt(max(b1**2 - 4 * c1, 0.0))) / 2
    b2 = 2 * (height + width)
    c2 = (1 - MIN_OVERLAP) * width * height
    r2 = (b2 + np.sqrt(max(b2**2 - 16 * c2, 0.0))) / 2
    a3 = 4 * MIN_OVERLAP
    b3 = -2 * MIN_OVERLAP * (height + width)
    c3 = (MIN_OVERLAP - 1) * width * height
    r3 = (b3 + np.sqrt(max(b3**2 - 4 * a3 * c3, 0.0))) / 2
    return float(max(1.0, min(r1, r2, r3)))


def _draw_gaussian(heatmap: np.ndarray, *, row: int, col: int, radius: int) -> None:
    """Raise a [G, G] class channel to a Gaussian of peak 1 at (row, col), clipped at the grid's edges."""
    sigma = (2 * radius + 1) / 6
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    gaussian = np.exp(-(offsets[:, None] ** 2 + offsets[None, :] ** 2) / (2 * sigma**2))
    gaussian[gaussian < np.finfo(np.float32).eps * gaussian.max()] = 0.0
    height, width = heatmap.shape
    left, right = min(col, radius), min(width - col, radius + 1)
    top, bottom = min(row, radius), min(height - row, radius + 1)
    patch = heatmap[row - top : row + bottom, col - left : col + right]
    np.maximum(patch, gaussian[radius - top : radius + bottom, radius - left : radius + right], out=patch)


def render(boxes: np.ndarray, labels: np.ndarray, *, grid_size: int, num_classes: int) -> Targets[np.ndarray]:
    """Targets for a scene's boxes [N, 4] and class labels [N].

    Objects whose centers fall in the same cell share its heatmap peak, and the
    last of them sets the cell's size and offset; CenterNet instead keeps every
    object's size and offset.
    """
    heatmap = np.zeros((grid_size, grid_size, num_classes), dtype=np.float32)
    size = np.zeros((grid_size, grid_size, 2), dtype=np.float32)
    offset = np.zeros((grid_size, grid_size, 2), dtype=np.float32)
    mask = np.zeros((grid_size, grid_size), dtype=np.float32)
    for box, label in zip(boxes, labels, strict=True):
        ymin, xmin, ymax, xmax = (float(value) * grid_size for value in box)
        height, width = ymax - ymin, xmax - xmin
        if height <= 0 or width <= 0:
            continue
        center_row, center_col = (ymin + ymax) / 2, (xmin + xmax) / 2
        row, col = min(int(center_row), grid_size - 1), min(int(center_col), grid_size - 1)
        _draw_gaussian(heatmap[:, :, label], row=row, col=col, radius=int(gaussian_radius(height, width)))
        size[row, col] = (height, width)
        offset[row, col] = (center_row - row, center_col - col)
        mask[row, col] = 1.0
    return Targets(heatmap=heatmap, size=size, offset=offset, mask=mask)
