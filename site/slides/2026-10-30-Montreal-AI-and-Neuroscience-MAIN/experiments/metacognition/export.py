"""The model's own uncertainty about what is where, for #uncertainty. Replays the street EG-C2F rollout
(ADE_val_00001780, the released CanViT-B with its 64 x 64 ADE20K probe, as site/record_bundles.sh records street-egc2f)
for its first five glimpses (the full scene, then the four quadrants in EG-C2F's order), and writes uncolored data to
<out>/:
  entropy_t<t>.png   16-bit gray: each canvas cell's predictive entropy divided by log(150), after glimpse t
  cells.json         after glimpse 0: the most and least certain cells (by entropy) with their top class probabilities
  tiles.json         after each glimpse: each quadrant's mean entropy (nats), and the quadrant EG-C2F visits next"""

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from canvit_pytorch.benchmarks.ade20k import CLASS_NAMES, dataset_root
from canvit_pytorch.policies import make_policy
from canvit_pytorch.policies.entropy import predictive_entropy, tile_masks
from canvit_pytorch.policies.quadtree import quadtree_level
from canvit_pytorch.preprocess import preprocess
from canvit_pytorch.viz.record import record
from canvit_pytorch.viz.released_model import CANVAS_GRID_SIZE, GLIMPSE_SIZE_PX, SCENE_SIZE_PX, load_released_segmenter
from PIL import Image

from experiments import logs
from experiments.outputs import DECK_DATA, SITE

log = logging.getLogger(__name__)

TOP_K = 6


@dataclass(frozen=True)
class Config:
    image_id: str = "ADE_val_00001780"
    num_glimpses: int = 5
    bundle: Path = SITE / "data/street-egc2f"
    """the recorded EG-C2F rollout the replay must match, viewpoint for viewpoint"""
    out: Path = DECK_DATA / "metacognition/street"
    device: str = "mps"


def main(cfg: Config) -> None:
    device = torch.device(cfg.device)
    model = load_released_segmenter(scene_size_px=SCENE_SIZE_PX, canvas_grid_size=CANVAS_GRID_SIZE, device=device).model
    image = preprocess(SCENE_SIZE_PX)(Image.open(dataset_root() / "images/validation" / f"{cfg.image_id}.jpg").convert("RGB"))
    image = image.unsqueeze(0).to(device)
    policy = make_policy("entropy_coarse_to_fine", batch_size=1, device=device, num_glimpses=cfg.num_glimpses,
                         canvas_grid_size=CANVAS_GRID_SIZE, canvas_logits=model.logits)
    rollout = record(model, image, policy, num_glimpses=cfg.num_glimpses, canvas_grid_size=CANVAS_GRID_SIZE,
                     glimpse_size_px=GLIMPSE_SIZE_PX, capture_writes=False, annotation=None)

    # The replay must take the bundle's viewpoints, or the slide would explain a different rollout.
    bundle = json.loads((cfg.bundle / "manifest.json").read_text())["glimpses"]
    for glimpse in rollout.glimpses:
        recorded = bundle[glimpse.t]["viewpoint"]
        row, col = glimpse.viewpoint.centers[0].tolist()
        scale = float(glimpse.viewpoint.scales[0])
        assert np.allclose([row, col, scale], [recorded["row"], recorded["col"], recorded["scale"]], atol=1e-4), \
            (glimpse.t, row, col, scale, recorded)

    cfg.out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rollout.scene).save(cfg.out / "scene.png")
    quadrants = quadtree_level(1)
    masks = tile_masks(quadrants, canvas_grid_size=CANVAS_GRID_SIZE, device=torch.device("cpu")).float()
    tiles = []
    for glimpse in rollout.glimpses:
        entropy = predictive_entropy(torch.from_numpy(glimpse.logits)[None])[0]  # [G, G] nats
        fraction = (entropy / math.log(len(CLASS_NAMES))).clamp(0, 1).numpy()
        Image.fromarray((fraction * 65535).round().astype(np.uint16)).save(cfg.out / f"entropy_t{glimpse.t}.png")
        means = ((entropy[None] * masks).sum(dim=(1, 2)) / masks.sum(dim=(1, 2))).tolist()
        tiles.append({"after_glimpse": glimpse.t, "quadrants": [{"row": q.row, "col": q.col, "scale": q.scale,
                                                                 "mean_entropy_nats": round(m, 4)} for q, m in zip(quadrants, means)]})
        log.info("t=%d entropy mean %.3f nats, quadrant means %s", glimpse.t, float(entropy.mean()), [round(m, 3) for m in means])

    # Which quadrant EG-C2F visited after each glimpse of level 1, from the recorded viewpoints.
    for t in range(1, cfg.num_glimpses):
        row, col = rollout.glimpses[t].viewpoint.centers[0].tolist()
        tiles[t - 1]["next"] = {"row": row, "col": col}
    (cfg.out / "tiles.json").write_text(json.dumps({"image_id": cfg.image_id, "entropy_unit": "nats",
                                                    "max_entropy_nats": math.log(len(CLASS_NAMES)), "after": tiles}, indent=1))

    first = torch.from_numpy(rollout.glimpses[0].logits)
    probs = torch.softmax(first, dim=0)  # [C, G, G]
    entropy = predictive_entropy(first[None])[0]
    cells = []
    for kind, flat in (("certain", int(entropy.argmin())), ("uncertain", int(entropy.argmax()))):
        r, c = divmod(flat, CANVAS_GRID_SIZE)
        top = torch.topk(probs[:, r, c], TOP_K)
        cells.append({"kind": kind, "row": r, "col": c, "grid": CANVAS_GRID_SIZE,
                      "entropy_nats": round(float(entropy[r, c]), 4),
                      "top": [{"class": CLASS_NAMES[i], "p": round(float(v), 4)} for v, i in zip(top.values.tolist(), top.indices.tolist())],
                      "p_rest": round(float(1 - top.values.sum()), 4)})
        log.info("%s cell (%d, %d): entropy %.3f, top %s", kind, r, c, float(entropy[r, c]), cells[-1]["top"][:3])
    (cfg.out / "cells.json").write_text(json.dumps({"image_id": cfg.image_id, "after_glimpse": 0, "cells": cells}, indent=1))


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
