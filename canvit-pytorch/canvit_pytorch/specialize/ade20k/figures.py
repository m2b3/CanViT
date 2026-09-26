"""Figures of a probe's predictions, logged to Comet while it trains.

One row per image: the image, its labels, then for the first and the last feature map the predicted labels,
where they are correct, and the features' PCA (canvit_pytorch.viz.pca, fitted on the last map).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from torch import Tensor

from canvit_pytorch.benchmarks.ade20k import IGNORE_LABEL, NUM_CLASSES
from canvit_pytorch.preprocess import imagenet_denormalize
from canvit_pytorch.viz import pca

LABEL_COLORS = np.random.RandomState(42).randint(0, 255, (NUM_CLASSES + 1, 3), dtype=np.uint8)
LABEL_COLORS[NUM_CLASSES] = 0  # ignored pixels are black
CORRECT, WRONG, IGNORED = (0, 200, 0), (200, 0, 0), (128, 128, 128)


def colorize(labels: NDArray[np.int64]) -> NDArray[np.uint8]:
    return LABEL_COLORS[np.where(labels == IGNORE_LABEL, NUM_CLASSES, labels)]


def correctness(predictions: NDArray[np.int64], labels: NDArray[np.int64]) -> NDArray[np.uint8]:
    colors = np.empty((*labels.shape, 3), dtype=np.uint8)
    labeled = labels != IGNORE_LABEL
    colors[labeled & (predictions == labels)] = CORRECT
    colors[labeled & (predictions != labels)] = WRONG
    colors[~labeled] = IGNORED
    return colors


def prediction_figure(*, images: Tensor, labels: Tensor, maps: list[Tensor], predictions: list[Tensor]) -> Figure:
    """images [B, 3, S, S] ImageNet-normalized, labels [B, S, S], maps [B, H, W, D] and predictions [B, S, S] per map."""
    shown = sorted({0, len(maps) - 1})
    map_height, map_width = maps[0].shape[1:3]
    num_rows, num_columns = images.shape[0], 2 + 3 * len(shown)
    figure, axes = plt.subplots(num_rows, num_columns, figsize=(2.5 * num_columns, 2.5 * num_rows), squeeze=False)
    for row in range(num_rows):
        label_map = labels[row].cpu().numpy()
        tokens = {t: maps[t][row].float().flatten(0, 1).cpu().numpy() for t in shown}
        basis = pca.fit_pca(tokens[shown[-1]])
        projections = {t: pca.project(basis, tokens[t]) for t in shown}
        limits = pca.color_limits(projections[shown[-1]])
        panels = [("image", imagenet_denormalize(images[row]).permute(1, 2, 0).cpu().numpy()), ("labels", colorize(label_map))]
        for t in shown:
            prediction = predictions[t][row].cpu().numpy()
            panels += [
                (f"prediction t{t}", colorize(prediction)),
                (f"correct t{t}", correctness(prediction, label_map)),
                (f"PCA t{t}", pca.to_rgb(projections[t], limits).reshape(map_height, map_width, 3)),
            ]
        for axis, (title, panel) in zip(axes[row], panels, strict=True):
            axis.imshow(panel)
            axis.set_title(title if row == 0 else "")
            axis.axis("off")
    figure.tight_layout()
    return figure
