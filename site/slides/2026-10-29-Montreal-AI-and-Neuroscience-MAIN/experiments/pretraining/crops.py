"""The glimpses of export.py's rollouts as images, for the static "Pretraining policies" slide: for each scene and
pretraining policy, glimpse t sampled from the 512 px scene at its viewpoint as the model receives it
(sample_at_viewpoint, 128 px), written as <dir>/crops/<scene>-<policy>-<t>.png, t from 1. Reads <dir>/rollouts.json."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from canvit_pytorch import Viewpoint, sample_at_viewpoint
from canvit_pytorch.viz.released_model import GLIMPSE_SIZE_PX
from PIL import Image

from experiments.outputs import DECK_DATA


@dataclass(frozen=True)
class Config:
    dir: Path = DECK_DATA / "pretraining"


def main(cfg: Config) -> None:
    record = json.loads((cfg.dir / "rollouts.json").read_text())
    (cfg.dir / "crops").mkdir(exist_ok=True)
    for scene in record["scenes"]:
        image = np.array(Image.open(cfg.dir / scene["image"]).convert("RGB"))
        assert image.shape[0] == image.shape[1], f"{scene['image']} is not square"
        spatial = torch.from_numpy(image).permute(2, 0, 1)[None].float() / 255
        for policy, viewpoints in scene["rollouts"].items():
            for t, (row, col, scale) in enumerate(viewpoints, start=1):
                viewpoint = Viewpoint(centers=torch.tensor([[row, col]]), scales=torch.tensor([scale]))
                glimpse = sample_at_viewpoint(spatial=spatial, viewpoint=viewpoint, glimpse_size_px=GLIMPSE_SIZE_PX)[0]
                Image.fromarray((glimpse.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(
                    cfg.dir / "crops" / f"{scene['name']}-{policy}-{t}.png")


if __name__ == "__main__":
    main(tyro.cli(Config))
