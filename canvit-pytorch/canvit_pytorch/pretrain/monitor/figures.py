"""A validation rollout as a figure, one row per glimpse.

Figures are built on matplotlib's Figure directly, without pyplot's global
registry, so a long run neither leaks figures nor depends on a display backend.
"""

import io
import math

import matplotlib
import numpy as np
import torch.nn.functional as F
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from numpy.typing import NDArray
from torch import Tensor

from canvit_pytorch.preprocess import imagenet_denormalize
from canvit_pytorch.viewpoint import Viewpoint, crop_box_px, sample_at_viewpoint
from canvit_pytorch.viz.pca import PCABasis, color_limits, fit_pca, project, to_rgb

COLUMNS = ("Viewpoints", "Glimpse", "Teacher", "Prediction", "Canvas", "1 − cos")


def _pca_image(tokens: NDArray[np.floating], basis: PCABasis, reference: NDArray[np.floating]) -> NDArray[np.uint8]:
    """tokens colored by the basis, with the color range of the reference tokens."""
    grid = math.isqrt(tokens.shape[0])
    return to_rgb(project(basis, tokens), color_limits(project(basis, reference))).reshape(grid, grid, 3)


def _rgb(image: Tensor) -> NDArray[np.floating]:
    return imagenet_denormalize(image.float()).permute(1, 2, 0).cpu().numpy()


def rollout_figure(
    *,
    image: Tensor,
    viewpoints: list[Viewpoint],
    glimpse_size_px: int,
    teacher_patches: Tensor,
    predicted_patches: list[Tensor],
    canvas_patches: list[Tensor],
) -> Figure:
    """One scene's rollout. Row t shows, after glimpse t: the viewpoints so far, the glimpse, PCA images of the
    teacher's patch features, of CanViT's prediction of them (in the teacher's basis and colors) and of the
    canvas, and the prediction's cosine distance to the teacher at each position.

    image: [3, H, W], ImageNet-normalized; viewpoints[t]: batch of one; teacher_patches and predicted_patches[t]:
    [G*G, D]; canvas_patches[t]: [G*G, canvas_dim].
    """
    num_glimpses = len(viewpoints)
    assert len(predicted_patches) == len(canvas_patches) == num_glimpses
    size_px = image.shape[-1]
    teacher = teacher_patches.float().cpu()
    teacher_basis = fit_pca(teacher.numpy())
    grid = math.isqrt(teacher.shape[0])
    colors = matplotlib.colormaps["viridis"](np.linspace(0, 1, num_glimpses))
    boxes = [crop_box_px(v, image_size_px=size_px)[0].tolist() for v in viewpoints]

    fig = Figure(figsize=(2.2 * len(COLUMNS), 2.2 * num_glimpses), layout="constrained")
    axes = fig.subplots(num_glimpses, len(COLUMNS), squeeze=False)
    for t, row in enumerate(axes):
        row[0].imshow(_rgb(image), extent=(0, size_px, size_px, 0))
        for i, (top, left, bottom, right) in enumerate(boxes[: t + 1]):
            row[0].add_patch(Rectangle(
                (left, top), right - left, bottom - top,
                fill=False, linewidth=1.5, edgecolor=colors[i], alpha=1.0 if i == t else 0.4,
            ))
        glimpse = sample_at_viewpoint(spatial=image[None].float(), viewpoint=viewpoints[t], glimpse_size_px=glimpse_size_px)
        row[1].imshow(_rgb(glimpse[0]))
        predicted = predicted_patches[t].float().cpu()
        canvas = canvas_patches[t].float().cpu().numpy()
        row[2].imshow(_pca_image(teacher.numpy(), teacher_basis, teacher.numpy()))
        row[3].imshow(_pca_image(predicted.numpy(), teacher_basis, teacher.numpy()))
        row[4].imshow(_pca_image(canvas, fit_pca(canvas), canvas))
        distance = 1 - F.cosine_similarity(predicted, teacher, dim=-1)
        fig.colorbar(row[5].imshow(distance.view(grid, grid).numpy(), cmap="magma", vmin=0, vmax=1), ax=row[5])
        row[0].set_ylabel(f"t = {t}")
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
    for ax, title in zip(axes[0], COLUMNS, strict=True):
        ax.set_title(title)
    return fig


def figure_png(fig: Figure) -> bytes:
    with io.BytesIO() as buffer:
        fig.savefig(buffer, format="png", dpi=100)
        return buffer.getvalue()
