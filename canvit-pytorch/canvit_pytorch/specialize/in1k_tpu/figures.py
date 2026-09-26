"""Comet figures of validation predictions: a glimpse beside the classifier's top-5 classes."""

import io
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch import Tensor

from canvit_pytorch.benchmarks.imagenet import CLASS_NAMES
from canvit_pytorch.preprocess import imagenet_denormalize

if TYPE_CHECKING:
    from comet_ml import CometExperiment

NUM_FIGURES = 4
TOP_K = 5


def log_validation_predictions(
    *, experiment: "CometExperiment", step: int, glimpses: Tensor, logits: Tensor, labels: Tensor,
) -> None:
    """One figure per sample for the first NUM_FIGURES samples: glimpses [B, 3, g, g], logits [B, 1000], labels [B]."""
    probabilities = F.softmax(logits, dim=-1)
    for i in range(min(NUM_FIGURES, glimpses.shape[0])):
        label = int(labels[i].item())
        top = probabilities[i].topk(TOP_K)
        top_classes, top_probabilities = top.indices.tolist(), top.values.tolist()
        correct = top_classes[0] == label

        figure, (image_axis, bar_axis) = plt.subplots(1, 2, figsize=(8, 3), gridspec_kw={"width_ratios": [1, 1.5]})
        image_axis.imshow(imagenet_denormalize(glimpses[i]).permute(1, 2, 0).numpy())
        image_axis.set_title(f"GT: {CLASS_NAMES[label]} ({label})", fontsize=9, color="green")
        image_axis.axis("off")
        rows = np.arange(TOP_K)
        bar_axis.barh(rows, top_probabilities, color=["green" if c == label else "steelblue" for c in top_classes])
        bar_axis.set_yticks(rows)
        bar_axis.set_yticklabels([f"{CLASS_NAMES[c][:25]} ({c})" for c in top_classes], fontsize=8)
        bar_axis.set_xlim(0, 1)
        bar_axis.invert_yaxis()
        bar_axis.set_title(f"{'CORRECT' if correct else 'WRONG'} (p={top_probabilities[0]:.2f})",
                           fontsize=9, color="green" if correct else "red")
        plt.tight_layout()

        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        buffer.seek(0)
        experiment.log_image(buffer, name=f"val/sample_{i}", step=step)
        plt.close(figure)
