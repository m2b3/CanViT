"""Numerical metrics for visualization."""

import numpy as np
from numpy.typing import NDArray


def cosine_dissimilarity(a: NDArray, b: NDArray) -> NDArray:
    """1 - cosine_similarity along last axis; [N, D] inputs → [N] in [0, 2]."""
    dot = (a * b).sum(axis=-1)
    norm_a = np.linalg.norm(a, axis=-1)
    norm_b = np.linalg.norm(b, axis=-1)
    cos_sim = dot / (norm_a * norm_b + 1e-8)
    return 1 - cos_sim
