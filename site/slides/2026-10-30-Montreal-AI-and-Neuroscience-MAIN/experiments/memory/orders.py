"""The same three glimpses in all six orders, for chosen sequences of the sweep: does the example depend on the
left-to-right order? Prints, per order, p over each object after the third glimpse, kept and reset."""

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from canvit_pytorch.viz.released_model import CANVAS_GRID_SIZE, load_released_segmenter

from experiments import logs
from experiments.ade20k import SCENE_PX, load
from experiments.memory.definition import SCORES, Geometry, candidates, rollouts, scores, targets
from experiments.memory.export import find
from experiments.outputs import WORK


@dataclass(frozen=True)
class Config:
    ids: tuple[str, ...]
    """IMAGE_ID:CLASS#COMPONENT,... as rank.py prints them"""
    sweep: Path = WORK / "memory/sweep.json"
    device: str = "mps"


@torch.inference_mode()
def main(cfg: Config) -> None:
    sweep = json.loads(cfg.sweep.read_text())
    geometry = Geometry(**sweep["geometry"])
    device = torch.device(cfg.device)
    model = load_released_segmenter(scene_size_px=SCENE_PX, canvas_grid_size=CANVAS_GRID_SIZE, device=device).model
    prob = SCORES.index("prob")
    for wanted in cfg.ids:
        row = find(sweep["sequences"], wanted)
        image, labels = load(row["image_id"])
        (c,) = [c for c in candidates(row["image_id"], labels, geometry)[0]
                if [(o.cls, o.component) for o in c.objs] == [(o["cls"], o["component"]) for o in row["objects"]]]
        orders = list(itertools.permutations(range(len(c.objs))))
        sequences = [tuple(c.viewpoints[i] for i in order) for order in orders]
        logits = rollouts(model, image[None].repeat(len(orders), 1, 1, 1).to(device), sequences)
        scene = targets(c, device)  # objects in left-to-right order, whatever the glimpse order
        kept = scores(logits["kept"][-1], [scene] * len(orders))[:, :, prob]
        reset = scores(logits["reset"][-1], [scene] * len(orders))[:, :, prob]
        names = [o.name for o in c.objs]
        print(f"{wanted}  (objects left to right: {', '.join(names)}; p after the third glimpse, kept | reset)")
        for n, order in enumerate(orders):
            print(f"  order {' -> '.join(names[i] for i in order):<40} kept {np.round(kept[n], 2)} | reset "
                  f"{np.round(reset[n], 2)}", flush=True)


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
