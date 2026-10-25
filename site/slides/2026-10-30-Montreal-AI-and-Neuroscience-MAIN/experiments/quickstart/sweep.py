"""A scene and a second glimpse for the quickstart slide, whose code takes the full scene, then one more glimpse: screen
ADE20K validation scenes for a second glimpse that visibly adds something. For each scene, after the full-scene glimpse,
every viewpoint on a grid of centers and scales is tried; a viewpoint scores the canvas cells of the single object class
(not stuff, ade20k.STUFF) that it turns from wrong to right, minus every cell it turns from right to wrong (against the
scene's annotation sampled at the 64 x 64 canvas cells). Writes one JSON line per scene, its best viewpoints first, as it
goes; the slide's scene is then chosen by eye among the best."""

import itertools
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from canvit_pytorch import Viewpoint, sample_at_viewpoint
from canvit_pytorch.benchmarks.ade20k import CLASS_NAMES, IGNORE_LABEL
from canvit_pytorch.viz.released_model import CANVAS_GRID_SIZE, GLIMPSE_SIZE_PX, load_released_segmenter

from experiments import logs
from experiments.ade20k import SCENE_PX, STUFF_IDS, load, validation_ids
from experiments.outputs import WORK

log = logging.getLogger(__name__)

BATCH = 32
KEEP = 5


@dataclass(frozen=True)
class Config:
    scenes: tuple[str, ...] = ()
    """ADE20K validation image ids; with --sample, a random sample of the validation set instead"""
    sample: int = 0
    seed: int = 0
    scales: tuple[float, ...] = (0.25, 0.33, 0.5)
    step: float = 0.125
    """spacing of candidate centers along each axis"""
    out: Path = WORK / "quickstart/sweep.jsonl"
    device: str = "mps"

    @property
    def candidates(self) -> list[tuple[float, float, float]]:
        return [(float(r), float(c), s) for s in self.scales
                for r, c in itertools.product(np.arange(-1 + s, 1 - s + 1e-6, self.step), repeat=2)]


def score(labels: np.ndarray, first: np.ndarray, truth: np.ndarray) -> dict:
    labeled = truth != IGNORE_LABEL
    fixed = (labels == truth) & (first != truth) & labeled
    broken = int(((labels != truth) & (first == truth) & labeled).sum())
    gained = {int(k): int((fixed & (truth == k)).sum()) for k in np.unique(truth[fixed]) if int(k) not in STUFF_IDS}
    best = max(gained.items(), key=lambda kv: kv[1], default=(None, 0))
    return {"score": best[1] - broken, "object": CLASS_NAMES[best[0]] if best[0] is not None else None,
            "object_cells": best[1], "broken": broken}


@torch.inference_mode()
def main(cfg: Config) -> None:
    assert bool(cfg.scenes) != bool(cfg.sample), "pass --scenes or --sample"
    scenes = list(cfg.scenes) or random.Random(cfg.seed).sample(validation_ids(), cfg.sample)
    device = torch.device(cfg.device)
    model = load_released_segmenter(scene_size_px=SCENE_PX, canvas_grid_size=CANVAS_GRID_SIZE, device=device).model
    candidates = cfg.candidates
    stride = SCENE_PX // CANVAS_GRID_SIZE
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    log.info("%d scenes, %d candidate second glimpses each, into %s", len(scenes), len(candidates), cfg.out)
    with cfg.out.open("w") as out:
        for i, image_id in enumerate(scenes):
            image, annotation = load(image_id)
            truth = annotation[stride // 2::stride, stride // 2::stride].astype(np.int64)  # each canvas cell's center
            scene = image.unsqueeze(0).to(device)
            state = model.init_state(batch_size=1, canvas_grid_size=CANVAS_GRID_SIZE)
            full = Viewpoint.full_scene(batch_size=1, device=device)
            logits, state = model(glimpse=sample_at_viewpoint(spatial=scene, viewpoint=full, glimpse_size_px=GLIMPSE_SIZE_PX),
                                  state=state, viewpoint=full)
            first = logits.argmax(1)[0].cpu().numpy()
            rows = []
            for j in range(0, len(candidates), BATCH):
                chunk = candidates[j:j + BATCH]
                n = len(chunk)
                viewpoint = Viewpoint(centers=torch.tensor([[r, c] for r, c, _ in chunk], dtype=torch.float32, device=device),
                                      scales=torch.tensor([s for *_, s in chunk], dtype=torch.float32, device=device))
                batch_state = type(state)(**{k: v.expand(n, *v.shape[1:]).clone() for k, v in vars(state).items()})
                logits, _ = model(glimpse=sample_at_viewpoint(spatial=scene.expand(n, -1, -1, -1), viewpoint=viewpoint,
                                                              glimpse_size_px=GLIMPSE_SIZE_PX),
                                  state=batch_state, viewpoint=viewpoint)
                for (r, c, s), labels in zip(chunk, logits.argmax(1).cpu().numpy()):
                    rows.append({"row": round(r, 3), "col": round(c, 3), "scale": s, **score(labels, first, truth)})
            rows.sort(key=lambda row: -row["score"])
            out.write(json.dumps({"image_id": image_id, "best": rows[:KEEP]}) + "\n")
            out.flush()
            top = rows[0]
            log.info("%d/%d %s: %+d (%s, %d cells; %d broken) at (%.2f, %.2f, %.2f)", i + 1, len(scenes), image_id,
                     top["score"], top["object"], top["object_cells"], top["broken"], top["row"], top["col"], top["scale"])


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
