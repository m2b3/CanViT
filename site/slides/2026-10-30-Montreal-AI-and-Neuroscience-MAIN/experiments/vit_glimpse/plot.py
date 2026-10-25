"""The images of "Processing a glimpse with a Vision Transformer", from export.py's npz, into <out>/: scene.png (the
photograph, downscaled), glimpse.png (the glimpse at its pixel size, upscaled with nearest neighbor), tokens.png (the
patch tokens projected on their first three principal components as colors, each clipped to its 2nd and 98th
percentiles, nearest neighbor) and meta.json (the glimpse's box as fractions of the scene, the token grid, the model)."""

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


@dataclass(frozen=True)
class Config:
    export: Path = WORK / "vit_glimpse/cat-glimpse.npz"
    out: Path = DECK_DATA / "vit-glimpse"


def main(cfg: Config) -> None:
    data = np.load(cfg.export)
    scene, glimpse, tokens = data["scene"], data["glimpse"], data["tokens"].astype(np.float64)
    grid = int(round(len(tokens) ** 0.5))
    assert grid * grid == len(tokens), f"{len(tokens)} tokens are not a square grid"
    cfg.out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(scene).resize((SHOWN_PX, round(SHOWN_PX * scene.shape[0] / scene.shape[1])), Image.Resampling.LANCZOS).save(cfg.out / "scene.png")
    Image.fromarray(glimpse).resize((SHOWN_PX, SHOWN_PX), Image.Resampling.NEAREST).save(cfg.out / "glimpse.png")
    rgb = colors(components(tokens, max(COMPONENTS) + 1)[:, list(COMPONENTS)], 2.0, 98.0).reshape(grid, grid, 3)
    Image.fromarray((rgb * 255).astype(np.uint8)).resize((SHOWN_PX, SHOWN_PX), Image.Resampling.NEAREST).save(cfg.out / "tokens.png")
    left, top, right, bottom = (int(v) for v in data["box"])
    h, w = scene.shape[:2]
    (cfg.out / "meta.json").write_text(json.dumps({
        "scene": "canvit-pytorch/test_data/Cat03.jpg (Fir0002/Flagstaffotos, CC BY-NC 3.0)",
        "box": {"top": top / h, "left": left / w, "size": (right - left) / w},
        "glimpse_px": int(glimpse.shape[0]), "token_grid": grid, "model": str(data["model"]),
        "pca": {"components": [c + 1 for c in COMPONENTS], "clip_percentiles": [2.0, 98.0]}}, indent=1))


if __name__ == "__main__":
    main(tyro.cli(Config))
