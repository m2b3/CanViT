"""The distillation slide's per-glimpse panels and loss, from an export (export.py); the model is loaded only for its
fitted teacher-feature standardizer. Writes <out>/<image>/<sequence>/view-<t>.png (the scene faded outside glimpse t,
its box in glimpse blue: what the model gets at t, in scene coordinates), loss-<t>.png (the pretraining loss per patch
after glimpse t on a fixed scale) and steps.json: each glimpse's viewpoint as the paper writes it, (x, y, scale) with
x the column and y the row in [-1, 1]; the pretraining patch loss after each glimpse (canvit_pytorch.pretrain.loss:
mean squared error between the standardized prediction and the standardized target, over patches and dimensions); and
the same loss for a prediction of zeros (the average target at every position), for scale."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch
import tyro
from canvit_pytorch.hub import repos
from canvit_pytorch.model.pretraining import CanViTForPretraining
from PIL import Image, ImageDraw

from experiments import logs
from experiments.outputs import DECK_DATA, WORK

log = logging.getLogger(__name__)

GLIMPSE_BLUE = "#2d6cdf"


@dataclass(frozen=True)
class Config:
    image: str
    sequence: str
    exports: Path = WORK / "distillation/exports"
    out: Path = DECK_DATA / "distillation"
    fade: float = 0.78
    """how far the unseen scene fades toward white"""
    cmap: str = "magma"


def view(scene: np.ndarray, viewpoint: np.ndarray, *, fade: float, width: int) -> Image.Image:
    side = scene.shape[0]
    row, col, s = viewpoint
    top, left, bottom, right = (round((v + 1) * side / 2) for v in (row - s, col - s, row + s, col + s))
    faded = (scene * (1 - fade) + 255 * fade).round().astype(np.uint8)
    faded[top:bottom, left:right] = scene[top:bottom, left:right]
    image = Image.fromarray(faded)
    ImageDraw.Draw(image).rectangle([left, top, right - 1, bottom - 1], outline=GLIMPSE_BLUE, width=width)
    return image


def main(cfg: Config) -> None:
    data = np.load(cfg.exports / f"{cfg.image}.npz")
    meta = json.loads(str(data["meta"]))
    assert meta["model"] == repos.FLAGSHIP, (meta["model"], repos.FLAGSHIP)
    standardizer = CanViTForPretraining.from_pretrained(repos.FLAGSHIP).teacher_patch_standardizer.eval()
    with torch.inference_mode():
        target = standardizer(torch.from_numpy(data["teacher_patches"].astype(np.float32)))  # [N, D]
        predicted = standardizer(torch.from_numpy(data[f"{cfg.sequence}/predicted_patches"].astype(np.float32)))  # [T, N, D]
    assert predicted.shape[1:] == target.shape, (predicted.shape, target.shape)
    per_patch = ((predicted - target) ** 2).mean(-1).numpy()  # [T, N]
    patch_loss = per_patch.mean(1)
    zeros_loss = float((target ** 2).mean())
    grid = meta["grid"]
    out = cfg.out / cfg.image / cfg.sequence
    out.mkdir(parents=True, exist_ok=True)
    top = float(np.quantile(per_patch[0], 0.98))  # one scale for every glimpse, so the maps compare
    for t, viewpoint in enumerate(data[f"{cfg.sequence}/viewpoints"]):
        view(data["scene"], viewpoint, fade=cfg.fade, width=max(4, data["scene"].shape[0] // 96)).save(out / f"view-{t}.png")
        colors = matplotlib.colormaps[cfg.cmap](np.clip(per_patch[t] / top, 0, 1).reshape(grid, grid))[..., :3]
        Image.fromarray((colors * 255).round().astype(np.uint8)).save(out / f"loss-{t}.png")
    (out / "steps.json").write_text(json.dumps({
        "viewpoints_x_y_scale": [[round(float(col), 4), round(float(row), 4), round(float(s), 4)]
                                 for row, col, s in data[f"{cfg.sequence}/viewpoints"]],
        "patch_loss": [round(float(v), 4) for v in patch_loss], "zeros_patch_loss": round(zeros_loss, 4),
        "loss_map_top": round(top, 4), "model": meta["model"]}, indent=1))
    log.info("%s: patch loss by glimpse %s, zeros %.3f", out, np.round(patch_loss, 3).tolist(), zeros_loss)


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
