"""PCA utilities for feature visualization."""

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA


def _pca_proj_to_rgb(proj: NDArray[np.floating], H: int, W: int) -> NDArray[np.floating]:
    """Convert PCA projection to RGB image via sigmoid. Clips to avoid overflow."""
    x = proj.reshape(H, W, 3) * 2.0
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def fit_pca(features: NDArray[np.floating], n_components: int = 12) -> PCA | None:
    """Fit PCA on [N, D] features; returns None if zero variance (e.g., constant init).

    Default 12 components supports offset viewing.
    """
    if features.var(axis=0).max() < 1e-5:
        return None
    n_components = min(n_components, features.shape[0], features.shape[1])
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(features)
    return pca


def pca_rgb(
    pca: PCA | None,
    features: NDArray[np.floating],
    H: int,
    W: int,
    normalize: bool = False,
    pc_offset: int = 0,
) -> NDArray[np.floating]:
    """Project [H*W, D] features to [H, W, 3] RGB via PCA. None pca → gray.

    pc_offset picks the component window (0=PC1-3, 1=PC2-4, ...).
    """
    if pca is None:
        return np.full((H, W, 3), 0.5, dtype=np.float32)
    proj = pca.transform(features)
    end = min(pc_offset + 3, proj.shape[1])
    start = max(0, end - 3)
    proj = proj[:, start:end]
    if proj.shape[1] < 3:
        pad = np.zeros((proj.shape[0], 3 - proj.shape[1]), dtype=proj.dtype)
        proj = np.concatenate([proj, pad], axis=1)
    if normalize:
        proj = proj / (proj.std() + 1e-8)
    return _pca_proj_to_rgb(proj, H, W)
