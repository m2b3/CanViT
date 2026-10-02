"""Every small object of ADE20K validation (definition.small_objects): how much of it CanViT's canvas labels as its
class after the full-scene glimpse alone (F), after F then a zoom on the object (FZ), and after F then F again (FF).
Writes the objects with their measures (metrics only), also every SAVE_EVERY images; summary.py aggregates them.

Per condition c: recall_c, the object's pixels labeled as its class; prob_c, the mean p(class) over the object;
fp_c, pixels of other labeled classes anywhere in the scene labeled as the object's class, over the object's area.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from canvit_pytorch.benchmarks.ade20k import IGNORE_LABEL
from canvit_pytorch.viz.released_model import CANVAS_GRID_SIZE, load_released_segmenter

from experiments import logs
from experiments.ade20k import SCENE_PX, load, validation_ids
from experiments.looking_closer.definition import CONDITIONS, SmallObject, canvit_logits, scene_maps, small_objects
from experiments.outputs import WORK

log = logging.getLogger(__name__)

BATCH = 16
SAVE_EVERY = 250  # images

Pending = tuple[SmallObject, np.ndarray, torch.Tensor, np.ndarray]  # object, its mask, the scene, its labels


@dataclass(frozen=True)
class Config:
    out: Path = WORK / "looking_closer/examples.json"
    device: str = "mps"


def measure(model, batch: list[Pending], device: torch.device) -> list[dict]:
    images = torch.stack([image for _, _, image, _ in batch]).to(device)
    classes = [obj.cls for obj, _, _, _ in batch]
    logits = canvit_logits(model, images, [obj.zoom for obj, _, _, _ in batch])
    maps = {cond: scene_maps(logits[cond], classes) for cond in CONDITIONS}
    rows = []
    for i, (obj, mask, _, labels) in enumerate(batch):
        others = (labels != obj.cls) & (labels != IGNORE_LABEL)
        row = asdict(obj)
        for cond, (pred, prob) in maps.items():
            row[f"recall_{cond}"] = float((pred[i][mask] == obj.cls).mean())
            row[f"prob_{cond}"] = float(prob[i][mask].mean())
            row[f"fp_{cond}"] = float(((pred[i] == obj.cls) & others).sum() / mask.sum())
        rows.append(row)
    return rows


@torch.inference_mode()
def main(cfg: Config) -> None:
    start = time.time()
    ids = validation_ids()
    device = torch.device(cfg.device)
    model = load_released_segmenter(scene_size_px=SCENE_PX, canvas_grid_size=CANVAS_GRID_SIZE, device=device).model
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    pending: list[Pending] = []
    for n, image_id in enumerate(ids, start=1):
        image, labels = load(image_id)
        pending += [(obj, mask, image, labels) for obj, mask in small_objects(image_id, labels)]
        while len(pending) >= BATCH or (n == len(ids) and pending):
            rows += measure(model, pending[:BATCH], device)
            pending = pending[BATCH:]
        if n % SAVE_EVERY == 0 or n == len(ids):
            cfg.out.write_text(json.dumps(rows, indent=1))
            means = {c: np.mean([r[f"recall_{c}"] for r in rows]) for c in CONDITIONS}
            log.info("%d/%d images, %d objects, %.0fs; mean recall F %.3f FZ %.3f FF %.3f", n, len(ids), len(rows),
                     time.time() - start, means["f"], means["fz"], means["ff"])
    log.info("done: %d objects in %.0fs -> %s", len(rows), time.time() - start, cfg.out)


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
