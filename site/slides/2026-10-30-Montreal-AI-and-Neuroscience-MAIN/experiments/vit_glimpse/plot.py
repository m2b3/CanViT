"""The images of the SR-RoPE slide (and, until 2026-10-27, of "Processing a glimpse with a Vision Transformer", whose
token views below no slide reads now), from export.py's npz, into
<out>/: scene.png (the 512 px scene), glimpse-<name>.png (each 128 px glimpse, upscaled with nearest neighbor so its
pixels show), tokens.png (the "vit" glimpse's patch tokens projected on their first three principal components as
colors, each clipped to its 2nd and 98th percentiles, nearest neighbor; tokens-<first>.png for each triplet of
components the slide cycles through, 1-3, 4-6, ..., tokens.png being the first) and meta.json (each glimpse's box as
fractions of the scene, the token grid, the model, the views cycled through)."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from PIL import Image

from experiments.foundation.pca_colors import CLIP_PERCENTILES, pca_cycle
from experiments.outputs import DECK_DATA, WORK

CYCLE = 8  # triplets of principal components shown in turn on the slide, as on the DINOv3 slide
SHOWN_PX = 512
GLIMPSES = ("vit", "zoom")


@dataclass(frozen=True)
class Config:
    export: Path = WORK / "vit_glimpse/cat-glimpses.npz"
    out: Path = DECK_DATA / "vit-glimpse"


def main(cfg: Config) -> None:
    data = np.load(cfg.export)
    tokens = data["tokens"].astype(np.float64)
    grid = round(len(tokens) ** 0.5)
    assert grid * grid == len(tokens), f"{len(tokens)} tokens are not a square grid"
    cfg.out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(data["scene"]).save(cfg.out / "scene.png")
    boxes = {}
    for name in GLIMPSES:
        glimpse = data[f"glimpse_{name}"]
        Image.fromarray(glimpse).resize((SHOWN_PX, SHOWN_PX), Image.Resampling.NEAREST).save(cfg.out / f"glimpse-{name}.png")
        row, col, scale = (float(v) for v in data[f"viewpoint_{name}"])
        boxes[name] = {"top": (row - scale + 1) / 2, "left": (col - scale + 1) / 2, "size": scale}
    cycle = pca_cycle(tokens, CYCLE)
    for k, (triplet, rgb) in enumerate(cycle):
        image = Image.fromarray((rgb.reshape(grid, grid, 3) * 255).astype(np.uint8)).resize((SHOWN_PX, SHOWN_PX), Image.Resampling.NEAREST)
        image.save(cfg.out / f"tokens-{triplet[0]}.png")
        if k == 0:
            image.save(cfg.out / "tokens.png")
    (cfg.out / "meta.json").write_text(json.dumps({
        "scene": "canvit-pytorch/test_data/Cat03.jpg (Fir0002/Flagstaffotos, CC BY-NC 3.0)",
        "boxes": boxes, "glimpse_px": int(data["glimpse_vit"].shape[0]), "token_grid": grid, "model": str(data["model"]),
        "clip_percentiles": list(CLIP_PERCENTILES), "feature_dim": int(tokens.shape[1]),
        "pca_cycle": [{"components": list(t), "image": f"tokens-{t[0]}.png"} for t, _ in cycle]}, indent=1))


if __name__ == "__main__":
    main(tyro.cli(Config))
