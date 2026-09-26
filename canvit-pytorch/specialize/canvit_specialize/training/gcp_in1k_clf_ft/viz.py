"""Comet visualization helpers for training."""

import logging

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger("viz")

_CLASS_NAMES: list[str] | None = None


def imagenet_class_names() -> list[str]:
    global _CLASS_NAMES
    if _CLASS_NAMES is None:
        from torchvision.models import ResNet18_Weights
        _CLASS_NAMES = ResNet18_Weights.DEFAULT.meta["categories"]
    return _CLASS_NAMES


def log_val_samples(
    *, exp, step: int, glimpses: torch.Tensor,
    logits: torch.Tensor, labels: torch.Tensor,
    imagenet_mean: tuple[float, ...], imagenet_std: tuple[float, ...],
) -> None:
    """Log top-5 prediction bar charts for a few val samples to Comet."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean = torch.tensor(imagenet_mean).view(3, 1, 1)
    std = torch.tensor(imagenet_std).view(3, 1, 1)
    probs = F.softmax(logits, dim=-1)
    class_names = imagenet_class_names()

    for i in range(min(4, glimpses.shape[0])):
        img = (glimpses[i] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        gt = labels[i].item()
        top5 = probs[i].topk(5)
        top5_idx = [top5.indices[j].item() for j in range(5)]
        top5_prob = [top5.values[j].item() for j in range(5)]
        top5_names = [class_names[idx] for idx in top5_idx]

        fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(8, 3),
                                              gridspec_kw={"width_ratios": [1, 1.5]})
        ax_img.imshow(img)
        ax_img.set_title(f"GT: {class_names[gt]} ({gt})", fontsize=9, color="green")
        ax_img.axis("off")

        colors = ["green" if idx == gt else "steelblue" for idx in top5_idx]
        y = np.arange(5)
        ax_bar.barh(y, top5_prob, color=colors)
        ax_bar.set_yticks(y)
        ax_bar.set_yticklabels([f"{n[:25]} ({idx})" for n, idx in zip(top5_names, top5_idx)], fontsize=8)
        ax_bar.set_xlim(0, 1)
        ax_bar.invert_yaxis()
        is_correct = top5_idx[0] == gt
        ax_bar.set_title(f"{'CORRECT' if is_correct else 'WRONG'} (p={top5_prob[0]:.2f})",
                         fontsize=9, color="green" if is_correct else "red")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        exp.log_image(buf, name=f"val/sample_{i}", step=step)
        plt.close(fig)
