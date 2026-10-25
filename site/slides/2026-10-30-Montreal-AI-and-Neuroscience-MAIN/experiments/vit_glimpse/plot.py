"""The images of "Processing a glimpse with a Vision Transformer" and of the SR-RoPE slide, from export.py's npz, into
<out>/: scene.png (the 512 px scene), glimpse-<name>.png (each 128 px glimpse, upscaled with nearest neighbor so its
pixels show), tokens.png (the "vit" glimpse's patch tokens projected on their first three principal components as
colors, each clipped to its 2nd and 98th percentiles, nearest neighbor) and meta.json (each glimpse's box as fractions
of the scene, the token grid, the model)."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from PIL import Image

from experiments.foundation.pca_colors import colors, components
from experiments.outputs import DECK_DATA, WORK

COMPONENTS = (0, 1, 2)
SHOWN_PX = 512
GLIMPSES = ("vit", "zoom")


@dataclass(frozen=True)
class Config:
    export: Path = WORK / "vit_glimpse/cat-glimpses.npz"
    out: Path = DECK_DATA / "vit-glimpse"


def main(cfg: Config) -> None:
    data = np.load(cfg.export)
    tokens = data["tokens"].astype(np.float64)
    grid = int(round(len(tokens) ** 0.5))
    assert grid * grid == len(tokens), f"{len(tokens)} tokens are not a square grid"
    cfg.out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(data["scene"]).save(cfg.out / "scene.png")
    boxes = {}
    for name in GLIMPSES:
        glimpse = data[f"glimpse_{name}"]
        Image.fromarray(glimpse).resize((SHOWN_PX, SHOWN_PX), Image.Resampling.NEAREST).save(cfg.out / f"glimpse-{name}.png")
        row, col, scale = (float(v) for v in data[f"viewpoint_{name}"])
        boxes[name] = {"top": (row - scale + 1) / 2, "left": (col - scale + 1) / 2, "size": scale}
    rgb = colors(components(tokens, max(COMPONENTS) + 1)[:, list(COMPONENTS)], 2.0, 98.0).reshape(grid, grid, 3)
    Image.fromarray((rgb * 255).astype(np.uint8)).resize((SHOWN_PX, SHOWN_PX), Image.Resampling.NEAREST).save(cfg.out / "tokens.png")
    (cfg.out / "meta.json").write_text(json.dumps({
        "scene": "canvit-pytorch/test_data/Cat03.jpg (Fir0002/Flagstaffotos, CC BY-NC 3.0)",
        "boxes": boxes, "glimpse_px": int(data["glimpse_vit"].shape[0]), "token_grid": grid, "model": str(data["model"]),
        "pca": {"components": [c + 1 for c in COMPONENTS], "clip_percentiles": [2.0, 98.0]}}, indent=1))


if __name__ == "__main__":
    main(tyro.cli(Config))
