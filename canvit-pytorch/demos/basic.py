"""The canvas across two viewing orders: the same viewpoints, overview first or details first.

Each row shows the scene with its viewpoints, then per glimpse the crop CanViT
saw and its canvas, colored by the paper's PCA protocol with one basis and one
color range for every canvas in the figure.

    uv run --extra demo python demos/basic.py
    uv run --extra demo python demos/basic.py --image path/to/image.jpg
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from matplotlib.patches import Rectangle
from PIL import Image

from canvit_pytorch import CanViTForPretraining, Viewpoint
from canvit_pytorch.episode import run_episode
from canvit_pytorch.hub.repos import (
    FLAGSHIP,
    RELEASED_CANVAS_GRID_SIZE,
    RELEASED_GLIMPSE_SIZE_PX,
    RELEASED_SCENE_SIZE_PX,
)
from canvit_pytorch.policies import FixedSequence
from canvit_pytorch.preprocess import imagenet_denormalize, preprocess
from canvit_pytorch.viewpoint import crop_box_px, sample_at_viewpoint
from canvit_pytorch.viz.pca import color_limits, fit_pca, project, to_rgb

Keypoint = tuple[float, float, float]  # (row, col, scale); (-1, -1) is the top-left corner

ORDERS: dict[str, list[Keypoint]] = {
    "Overview first": [(0.0, 0.0, 1.0), (-0.3, -0.3, 0.35), (-0.1, 0.15, 0.35), (0.2, 0.35, 0.3)],
    "Details first": [(0.2, 0.35, 0.3), (-0.1, 0.15, 0.35), (-0.3, -0.3, 0.35), (0.0, 0.0, 1.0)],
}
GLIMPSE_COLOR = "#4080d0"  # the paper figures' glimpse blue


@dataclass(frozen=True)
class Demo:
    image: Path = Path("test_data/Places365_IMG_9600.jpeg")
    model_repo: str = FLAGSHIP
    scene_size_px: int = RELEASED_SCENE_SIZE_PX
    canvas_grid_size: int = RELEASED_CANVAS_GRID_SIZE
    glimpse_size_px: int = RELEASED_GLIMPSE_SIZE_PX
    output: Path = Path("outputs/demo.png")


def viewpoint(keypoint: Keypoint) -> Viewpoint:
    row, col, scale = keypoint
    return Viewpoint(centers=torch.tensor([[row, col]]), scales=torch.tensor([scale]))


@torch.inference_mode()
def canvases_along(model: CanViTForPretraining, image: torch.Tensor, keypoints: list[Keypoint], demo: Demo) -> list[np.ndarray]:
    """[G*G, canvas_dim] canvas patches after each glimpse."""
    steps = run_episode(
        canvit=model.canvit, images=image, policy=FixedSequence([viewpoint(k) for k in keypoints]),
        num_glimpses=len(keypoints), glimpse_size_px=demo.glimpse_size_px,
        initial_state=model.init_state(batch_size=1, canvas_grid_size=demo.canvas_grid_size),
    )
    return [model.canvit.canvas_patches(step.state.canvas)[0].float().numpy() for step in steps]


def main(demo: Demo) -> None:
    model = CanViTForPretraining.from_pretrained(demo.model_repo).eval()
    image = preprocess(demo.scene_size_px)(Image.open(demo.image).convert("RGB"))
    assert isinstance(image, torch.Tensor)
    image = image.unsqueeze(0)
    canvases = {name: canvases_along(model, image, keypoints, demo) for name, keypoints in ORDERS.items()}

    everything = [c for rows in canvases.values() for c in rows]
    basis = fit_pca(*everything)
    limits = color_limits(*(project(basis, c) for c in everything))
    grid = demo.canvas_grid_size

    num_glimpses = len(next(iter(ORDERS.values())))
    fig, axes = plt.subplots(2 * len(ORDERS), num_glimpses + 1, figsize=(2.6 * (num_glimpses + 1), 5.4 * len(ORDERS)),
                             gridspec_kw={"hspace": 0.12, "wspace": 0.06})
    scene = imagenet_denormalize(image[0]).permute(1, 2, 0).clamp(0, 1).numpy()
    for row, (name, keypoints) in enumerate(ORDERS.items()):
        top, bottom = axes[2 * row], axes[2 * row + 1]
        bottom[0].remove()
        top[0].imshow(scene)
        top[0].set_title(name, fontweight="bold")
        for t, keypoint in enumerate(keypoints):
            y0, x0, y1, x1 = crop_box_px(viewpoint(keypoint), image_size_px=demo.scene_size_px)[0].tolist()
            top[0].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=GLIMPSE_COLOR, linewidth=2))
            top[0].text(x0 + 4, y0 + 18, f"t={t}", color="white", fontsize=9, fontweight="bold",
                        bbox={"facecolor": GLIMPSE_COLOR, "linewidth": 0, "pad": 1.5})
            glimpse = sample_at_viewpoint(spatial=image, viewpoint=viewpoint(keypoint), glimpse_size_px=demo.glimpse_size_px)
            top[t + 1].imshow(imagenet_denormalize(glimpse[0]).permute(1, 2, 0).clamp(0, 1).numpy())
            top[t + 1].set_title(f"glimpse t={t}", fontsize=9)
            bottom[t + 1].imshow(to_rgb(project(basis, canvases[name][t]), limits).reshape(grid, grid, 3), interpolation="nearest")
            bottom[t + 1].set_title(f"canvas after t={t}", fontsize=9)
        for ax in (*top, *bottom[1:]):
            ax.set_xticks([])
            ax.set_yticks([])
    demo.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(demo.output, dpi=150, bbox_inches="tight")
    print(f"Wrote {demo.output}")


if __name__ == "__main__":
    main(tyro.cli(Demo))
