"""The glimpses of export.py's rollouts as images, for the static "Pretraining policies" slide: for each scene and
pretraining policy, glimpse t cropped from the 512 px scene at its viewpoint and resized to the 128 px glimpse size
(as the model receives it), written as <dir>/crops/<scene>-<policy>-<t>.png, t from 1. Reads <dir>/rollouts.json."""

import json
from dataclasses import dataclass
from pathlib import Path

import tyro
from PIL import Image

from experiments.outputs import DECK_DATA

GLIMPSE_PX = 128


@dataclass(frozen=True)
class Config:
    dir: Path = DECK_DATA / "pretraining"


def main(cfg: Config) -> None:
    record = json.loads((cfg.dir / "rollouts.json").read_text())
    (cfg.dir / "crops").mkdir(exist_ok=True)
    for scene in record["scenes"]:
        image = Image.open(cfg.dir / scene["image"]).convert("RGB")
        side = image.size[0]
        assert image.size == (side, side), f"{scene['image']} is not square"
        for policy, viewpoints in scene["rollouts"].items():
            for t, (row, col, scale) in enumerate(viewpoints, start=1):
                box = tuple(round((v + 1) / 2 * side) for v in (col - scale, row - scale, col + scale, row + scale))
                image.crop(box).resize((GLIMPSE_PX, GLIMPSE_PX), Image.Resampling.BICUBIC).save(
                    cfg.dir / "crops" / f"{scene['name']}-{policy}-{t}.png")


if __name__ == "__main__":
    main(tyro.cli(Config))
