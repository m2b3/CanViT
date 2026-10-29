"""DINOv3 patch features as colors: projections on principal components of the LayerNormed features, each scaled to
[0, 1] between two of its percentiles."""

import numpy as np
from canvit_pytorch.viz.pca import layernorm


def components(features: np.ndarray, count: int) -> np.ndarray:
    """[N, count] projections on the first principal components of the LayerNormed features (sign as in viz.pca)."""
    normed = layernorm(features)
    centered = normed - normed.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    vt = vt[:count]
    vt *= np.sign(vt[np.arange(count), np.abs(vt).argmax(axis=1)])[:, None]
    return centered @ vt.T


def colors(projection: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    low, high = np.percentile(projection, low_pct, axis=0), np.percentile(projection, high_pct, axis=0)
    return np.clip((projection - low) / (high - low), 0, 1)


CLIP_PERCENTILES = (2.0, 98.0)


def pca_cycle(features: np.ndarray, views: int) -> list[tuple[tuple[int, int, int], np.ndarray]]:
    """The views a slide cycles through: for principal components 1-3, 4-6, ... (views triplets, numbered from 1),
    the [N, 3] colors of each patch, each component clipped to CLIP_PERCENTILES."""
    projection = components(features, 3 * views)
    return [((3 * k + 1, 3 * k + 2, 3 * k + 3), colors(projection[:, 3 * k:3 * k + 3], *CLIP_PERCENTILES)) for k in range(views)]
