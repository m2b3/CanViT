"""The paper's PCA visualization of token grids.

Protocol (paper appendix, "PCA Visualizations"): LayerNorm each token, project
onto three principal components, and map them to RGB with min-max scaling.
The components come from a deterministic full SVD with scikit-learn's sign
convention (the largest-magnitude loading of each component is positive), so
results match the scikit-learn implementation behind the paper's figures.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

LAYERNORM_EPS = 1e-5
RANGE_EPS = 1e-8
N_COMPONENTS = 3


def layernorm(tokens: NDArray[np.floating]) -> NDArray[np.float64]:
    """Zero mean, unit variance per token (over the last axis)."""
    tokens = np.asarray(tokens, dtype=np.float64)
    mean = tokens.mean(axis=-1, keepdims=True)
    var = tokens.var(axis=-1, keepdims=True)
    return (tokens - mean) / np.sqrt(var + LAYERNORM_EPS)


@dataclass(frozen=True)
class PCABasis:
    mean: NDArray[np.float64]  # [D], mean of the LayerNormed fitting tokens
    components: NDArray[np.float64]  # [N_COMPONENTS, D]


def fit_pca(*token_sets: NDArray[np.floating]) -> PCABasis:
    """Fit a basis on the LayerNormed union of one or more [N, D] token sets."""
    assert token_sets, "fit_pca needs at least one token set"
    normed = np.concatenate([layernorm(t) for t in token_sets], axis=0)
    mean = normed.mean(axis=0)
    _, _, vt = np.linalg.svd(normed - mean, full_matrices=False)
    vt = vt[:N_COMPONENTS]
    signs = np.sign(vt[np.arange(len(vt)), np.abs(vt).argmax(axis=1)])
    return PCABasis(mean=mean, components=vt * signs[:, None])


def project(basis: PCABasis, tokens: NDArray[np.floating]) -> NDArray[np.float64]:
    """[N, D] tokens -> [N, 3] coordinates in the basis."""
    return (layernorm(tokens) - basis.mean) @ basis.components.T


@dataclass(frozen=True)
class ColorLimits:
    low: NDArray[np.float64]  # [3]
    high: NDArray[np.float64]  # [3]


def color_limits(*projections: NDArray[np.floating]) -> ColorLimits:
    """Per-component min and max over one projection (the paper) or several (fixed limits)."""
    stacked = np.concatenate([np.asarray(p, dtype=np.float64) for p in projections], axis=0)
    return ColorLimits(low=stacked.min(axis=0), high=stacked.max(axis=0))


def to_rgb(projection: NDArray[np.floating], limits: ColorLimits) -> NDArray[np.uint8]:
    """[N, 3] coordinates -> [N, 3] uint8 colors, clipped to the limits."""
    scaled = (np.asarray(projection, dtype=np.float64) - limits.low) / (limits.high - limits.low + RANGE_EPS)
    return (np.clip(scaled, 0.0, 1.0) * 255).astype(np.uint8)
