"""What the sweep ranks scenes by and the plots report: how close CanViT's predicted teacher features are to the
teacher's, and how clearly the teacher's PCA colors separate the scene's annotated classes.

Colors follow the slide: one PCA basis and color limits, fitted on the teacher's patches alone (viz.pca, the paper's
protocol), applied to the teacher and to every prediction.
"""

import numpy as np
from canvit_pytorch.benchmarks.ade20k.classes import IGNORE_LABEL
from canvit_pytorch.viz.pca import ColorLimits, PCABasis, color_limits, fit_pca, layernorm, project, to_rgb

from experiments.distillation.sequences import GRID, PATCH_PX, SCENE_PX

MIN_CLASS_AREA = 0.02  # a class covering at least this share of the scene counts as a region
COLOR_SEPARATION = 0.25  # distance in [0, 1]³ RGB between class mean colors that counts as distinct


def patch_cosines(predicted: np.ndarray, teacher: np.ndarray) -> np.ndarray:
    """[..., N, D] against [N, D] -> [..., N] cosine similarity per patch."""
    predicted, teacher = predicted.astype(np.float64), teacher.astype(np.float64)
    return ((predicted / np.linalg.norm(predicted, axis=-1, keepdims=True))
            * (teacher / np.linalg.norm(teacher, axis=-1, keepdims=True))).sum(-1)


def teacher_pca(teacher: np.ndarray) -> tuple[PCABasis, ColorLimits]:
    basis = fit_pca(teacher)
    return basis, color_limits(project(basis, teacher))


def pca_colors(tokens: np.ndarray, basis: PCABasis, limits: ColorLimits) -> np.ndarray:
    """[N, D] -> [N, 3] uint8 RGB, as drawn."""
    return to_rgb(project(basis, tokens), limits)


def colors(tokens: np.ndarray, basis: PCABasis, limits: ColorLimits) -> np.ndarray:
    """pca_colors in [0, 1]."""
    return pca_colors(tokens, basis, limits).astype(np.float64) / 255


def explained_by_three_components(teacher: np.ndarray) -> float:
    """Share of the LayerNormed teacher patches' variance on the first three principal components."""
    normed = layernorm(teacher)
    singular = np.linalg.svd(normed - normed.mean(axis=0), compute_uv=False)
    return float((singular[:3] ** 2).sum() / (singular**2).sum())


def patch_labels(labels: np.ndarray) -> np.ndarray:
    """[SCENE_PX, SCENE_PX] class map -> [GRID * GRID] the class covering most of each patch when it covers more
    than half of it, else -1."""
    assert labels.shape == (SCENE_PX, SCENE_PX), labels.shape
    blocks = labels.reshape(GRID, PATCH_PX, GRID, PATCH_PX).transpose(0, 2, 1, 3).reshape(GRID * GRID, -1)
    out = np.full(GRID * GRID, -1)
    for i, block in enumerate(blocks):
        values, counts = np.unique(block, return_counts=True)
        k = counts.argmax()
        if values[k] != IGNORE_LABEL and counts[k] > block.size / 2:
            out[i] = values[k]
    return out


def class_separation(teacher_colors: np.ndarray, labels: np.ndarray) -> tuple[float, int, int]:
    """(share of the color variance between classes, over the patches patch_labels assigns a class; the classes
    covering MIN_CLASS_AREA of the scene and at least two such patches; how many of those have a mean color at
    least COLOR_SEPARATION from every other's)."""
    per_patch = patch_labels(labels)
    keep = per_patch >= 0
    x, y = teacher_colors[keep], per_patch[keep]
    grand = x.mean(axis=0)
    between = sum((y == c).sum() * ((x[y == c].mean(axis=0) - grand) ** 2).sum() for c in np.unique(y))
    eta2 = float(between / ((x - grand) ** 2).sum())
    values, counts = np.unique(labels[labels != IGNORE_LABEL], return_counts=True)
    regions = [c for c, n in zip(values, counts) if n >= MIN_CLASS_AREA * labels.size and (per_patch == c).sum() >= 2]
    means = np.array([teacher_colors[per_patch == c].mean(axis=0) for c in regions]).reshape(-1, 3)
    distances = np.linalg.norm(means[:, None] - means[None], axis=-1) + np.eye(len(means)) * 9
    separated = int((distances.min(axis=1) >= COLOR_SEPARATION).sum()) if len(means) > 1 else 0
    return eta2, len(regions), separated
