"""The data of #sr-rope, from export.py's npz, into <out>/: scene.png (the 512 px scene) and meta.json (each glimpse's
box as fractions of the scene, the glimpse's size and its token grid, the model)."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from PIL import Image

from experiments.outputs import DECK_DATA, WORK

GLIMPSES = ("vit", "zoom")


@dataclass(frozen=True)
class Config:
    export: Path = WORK / "vit_glimpse/cat-glimpses.npz"
    out: Path = DECK_DATA / "vit-glimpse"


def main(cfg: Config) -> None:
    data = np.load(cfg.export)
    tokens = data["tokens"]
    grid = round(len(tokens) ** 0.5)
    assert grid * grid == len(tokens), f"{len(tokens)} tokens are not a square grid"
    cfg.out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(data["scene"]).save(cfg.out / "scene.png")
    boxes = {}
    for name in GLIMPSES:
        row, col, scale = (float(v) for v in data[f"viewpoint_{name}"])
        boxes[name] = {"top": (row - scale + 1) / 2, "left": (col - scale + 1) / 2, "size": scale}
    (cfg.out / "meta.json").write_text(json.dumps({
        "scene": "canvit-pytorch/test_data/Cat03.jpg (Fir0002/Flagstaffotos, CC BY-NC 3.0)",
        "boxes": boxes, "glimpse_px": int(data["glimpse_vit"].shape[0]), "token_grid": grid, "model": str(data["model"]),
    }, indent=1))


if __name__ == "__main__":
    main(tyro.cli(Config))
