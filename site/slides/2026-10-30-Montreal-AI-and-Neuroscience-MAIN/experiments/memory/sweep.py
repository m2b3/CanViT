"""Every admissible three-object sequence of ADE20K validation (definition.py), scored after every glimpse with the
canvas kept and reset. Writes, also every SAVE_EVERY images, the geometry used and per sequence the scene's category
and short side, the objects in glimpse order, the viewpoints, foreign[j][i] (fraction of object i inside glimpse j),
and <score>[condition][t][i] for each of definition.SCORES."""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro
from canvit_pytorch.viz.released_model import CANVAS_GRID_SIZE, load_released_segmenter

from experiments import logs
from experiments.ade20k import SCENE_PX, load, scene_categories, validation_ids
from experiments.memory.definition import (
    CONDITIONS,
    SCORES,
    Candidate,
    Geometry,
    candidates,
    rollouts,
    scores,
    short_side_px,
    targets,
)
from experiments.outputs import WORK

log = logging.getLogger(__name__)

BATCH = 16
SAVE_EVERY = 250  # images
LOG_EVERY = 50  # images
DECIMALS = 4


@dataclass(frozen=True)
class Config:
    geometry: Geometry = field(default_factory=Geometry)
    out: Path = WORK / "memory/sweep.json"
    num_images: int = 2000
    """the first validation images, by id, to sweep (fewer for a smoke test)"""
    must_include: tuple[str, ...] = ()
    """keep only sequences with an object of one of these classes (every sequence when empty)"""
    min_short_side_px: int = 0
    """skip images whose photo's short side is smaller"""
    device: str = "mps"


@dataclass(frozen=True)
class Pending:
    image_id: str
    image: torch.Tensor
    candidate: Candidate


def measure(model, batch: list[Pending], categories: dict[str, str], short_sides: dict[str, int],
            device: torch.device) -> list[dict]:
    images = torch.stack([p.image for p in batch]).to(device)
    logits = rollouts(model, images, [p.candidate.viewpoints for p in batch])
    scenes = [targets(p.candidate, device) for p in batch]
    scored = {cond: np.stack([scores(step, scenes) for step in steps]) for cond, steps in logits.items()}  # [t, B, k, s]
    rows = []
    for b, p in enumerate(batch):
        c = p.candidate
        per_t = lambda cond, s: np.round(scored[cond][:, b, :, s], DECIMALS).tolist()  # noqa: E731  [t][i]
        rows.append({
            "image_id": p.image_id, "category": categories[p.image_id], "short_side_px": short_sides[p.image_id],
            "objects": [asdict(o) for o in c.objs],
            "viewpoints": [[round(v, 6) for v in vp] for vp in c.viewpoints],
            "foreign": c.foreign, "max_glimpse_overlap": round(c.max_glimpse_overlap, DECIMALS),
            **{name: {cond: per_t(cond, s) for cond in CONDITIONS} for s, name in enumerate(SCORES)},
        })
    return rows


def save(cfg: Config, rows: list[dict], num_tried: int, num_images: int) -> None:
    cfg.out.write_text(json.dumps({"geometry": asdict(cfg.geometry), "must_include": cfg.must_include,
                                   "min_short_side_px": cfg.min_short_side_px, "scene_px": SCENE_PX,
                                   "canvas_grid": CANVAS_GRID_SIZE, "num_images": num_images,
                                   "num_triples_tried": num_tried, "sequences": rows}))


@torch.inference_mode()
def main(cfg: Config) -> None:
    start = time.time()
    ids = validation_ids()[:cfg.num_images]
    categories = scene_categories()
    short_sides: dict[str, int] = {}
    device = torch.device(cfg.device)
    model = load_released_segmenter(scene_size_px=SCENE_PX, canvas_grid_size=CANVAS_GRID_SIZE, device=device).model
    log.info("model loaded in %.0fs; geometry %s", time.time() - start, cfg.geometry)
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    pending: list[Pending] = []
    num_tried = 0
    for n, image_id in enumerate(ids, start=1):
        short_sides[image_id] = short_side_px(image_id)
        if short_sides[image_id] >= cfg.min_short_side_px:
            image, labels = load(image_id)
            found, tried = candidates(image_id, labels, cfg.geometry)
            num_tried += tried
            pending += [Pending(image_id, image, c) for c in found
                        if not cfg.must_include or any(o.name in cfg.must_include for o in c.objs)]
        while len(pending) >= BATCH or (n == len(ids) and pending):
            rows += measure(model, pending[:BATCH], categories, short_sides, device)
            pending = pending[BATCH:]
        if n % SAVE_EVERY == 0 or n == len(ids):
            save(cfg, rows, num_tried, n)
        if n % LOG_EVERY == 0 or n == len(ids):
            kept = [r["prob"]["kept"][-1] for r in rows]
            reset = [r["prob"]["reset"][-1] for r in rows]
            log.info("%d/%d images, %d triples tried, %d sequences scored, %.0fs; mean p over the objects after the "
                     "last glimpse, kept %s reset %s", n, len(ids), num_tried, len(rows), time.time() - start,
                     np.round(np.mean(kept, axis=0), 3) if rows else "-", np.round(np.mean(reset, axis=0), 3) if rows else "-")
    log.info("done: %d sequences of %d triples in %.0fs -> %s", len(rows), num_tried, time.time() - start, cfg.out)


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
