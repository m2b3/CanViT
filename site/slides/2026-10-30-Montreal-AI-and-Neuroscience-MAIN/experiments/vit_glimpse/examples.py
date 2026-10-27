"""The examples of "Processing a glimpse with a Vision Transformer", from sweep.py's candidates: per example, into
<out>/<image_id>-<name>/, scene.png (the 512 px scene), glimpse.png (the 128 px glimpse, upscaled with nearest neighbor
so its pixels show), tokens-<first>.png for each triplet of principal components the slide cycles through (as
plot.py), and meta.json (the glimpse's box as fractions of the scene, the model, the views, the candidate's measures).
With --sheet, also a contact sheet of every example: scene with the glimpse's box, glimpse, PCs 1-3, measures."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from canvit_pytorch.teacher import TEACHER_REPO, load_teacher
from canvit_pytorch.viewpoint import sample_at_viewpoint
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from experiments.ade20k import load, pixels
from experiments.foundation.pca_colors import CLIP_PERCENTILES, pca_cycle
from experiments.glimpses import GLIMPSE_PX, viewpoints
from experiments.outputs import DECK_DATA, WORK

CYCLE = 8  # triplets of principal components shown in turn on the slide, as on the DINOv3 slide
SHOWN_PX = 512
# The slide's example first, then the alternatives, chosen by eye among sweep.py's best (TALK.md, "#vit-on-a-glimpse").
EXAMPLES = (
    "ADE_val_00001517:lamp:1", "ADE_val_00001639:plaything:4", "ADE_val_00001965:sofa:1",
    "ADE_val_00000308:chandelier:1", "ADE_val_00000121:door:2", "ADE_val_00001486:door:4", "ADE_val_00000273:cabinet:1",
    "ADE_val_00000167:cushion:3", "ADE_val_00000969:painting:1", "ADE_val_00001581:bicycle:2", "ADE_val_00000177:table:2",
)


@dataclass(frozen=True)
class Config:
    examples: tuple[str, ...] = EXAMPLES
    """image_id:name of sweep.py's candidates, or image_id:name:component where a class has two"""
    candidates: Path = WORK / "vit_glimpse/candidates.json"
    out: Path = DECK_DATA / "vit-glimpse-examples"
    sheet: Path | None = None
    device: str = "mps"


def find(candidates: list[dict], key: str) -> dict:
    image_id, name, *component = key.split(":")
    matches = [c for c in candidates if c["image_id"] == image_id and c["name"] == name
               and (not component or c["component"] == int(component[0]))]
    if len(matches) != 1:
        raise ValueError(f"{key}: {len(matches)} candidates match; name the component")
    return matches[0]


def main(cfg: Config) -> None:
    candidates = json.loads(cfg.candidates.read_text())["candidates"]
    chosen = [find(candidates, key) for key in cfg.examples]
    device = torch.device(cfg.device)
    teacher = load_teacher(TEACHER_REPO, device)
    shown = []
    for c in chosen:
        image, _ = load(c["image_id"])
        crop = sample_at_viewpoint(spatial=image[None].to(device), viewpoint=viewpoints([tuple(c["viewpoint"])], device),
                                   glimpse_size_px=GLIMPSE_PX)
        with torch.inference_mode():
            tokens = teacher(crop).patches[0].float().cpu().numpy().astype(np.float64)
        grid = round(len(tokens) ** 0.5)
        cycle = pca_cycle(tokens, CYCLE)
        out = cfg.out / f"{c['image_id']}-{c['name'].replace(' ', '-')}"
        out.mkdir(parents=True, exist_ok=True)
        scene, glimpse = pixels(image), pixels(crop[0])
        Image.fromarray(scene).save(out / "scene.png")
        Image.fromarray(glimpse).resize((SHOWN_PX, SHOWN_PX), Image.Resampling.NEAREST).save(out / "glimpse.png")
        for triplet, rgb in cycle:
            Image.fromarray((rgb.reshape(grid, grid, 3) * 255).astype(np.uint8)).resize(
                (SHOWN_PX, SHOWN_PX), Image.Resampling.NEAREST).save(out / f"tokens-{triplet[0]}.png")
        row, col, scale = c["viewpoint"]
        (out / "meta.json").write_text(json.dumps({
            "scene": f"ADE20K validation {c['image_id']}", "object": c["name"], "viewpoint": c["viewpoint"],
            "box": {"top": (row - scale + 1) / 2, "left": (col - scale + 1) / 2, "size": scale},
            "glimpse_px": GLIMPSE_PX, "token_grid": grid, "model": TEACHER_REPO, "clip_percentiles": list(CLIP_PERCENTILES),
            "feature_dim": int(tokens.shape[1]),
            "pca_cycle": [{"components": list(t), "image": f"tokens-{t[0]}.png"} for t, _ in cycle],
            "measures": {k: c[k] for k in ("pixel_auc", "token_auc", "token_fisher", "delta_e", "object_patches",
                                           "surrounding_patches")}}, indent=1))
        shown.append((c, scene, glimpse, cycle[0][1].reshape(grid, grid, 3)))
    if cfg.sheet:
        contact_sheet(shown, cfg.sheet)


def contact_sheet(shown: list, path: Path) -> None:
    columns = 2
    rows = (len(shown) + columns - 1) // columns
    fig, axes = plt.subplots(rows, 3 * columns, figsize=(6.6 * columns, 2.5 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, (c, scene, glimpse, tokens) in enumerate(shown):
        ax_scene, ax_glimpse, ax_tokens = axes[k // columns, 3 * (k % columns):3 * (k % columns) + 3]
        ax_scene.imshow(scene)
        row, col, s = c["viewpoint"]
        size = scene.shape[0]
        ax_scene.add_patch(Rectangle(((col - s + 1) * size / 2, (row - s + 1) * size / 2), s * size, s * size,
                                     fill=False, edgecolor="#3b6fd8", linewidth=1.5))
        ax_scene.set_title(f"{k + 1}. {c['image_id'][8:]} {c['name']}", fontsize=8, loc="left")
        ax_glimpse.imshow(glimpse)
        ax_glimpse.set_title(f"s = {s:.2f}", fontsize=8)
        ax_tokens.imshow(tokens, interpolation="nearest")
        ax_tokens.set_title(f"pixel AUC {c['pixel_auc']:.2f}, tokens {c['token_auc']:.2f}", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main(tyro.cli(Config))
