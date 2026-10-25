"""Viewpoints for the title slide's teaser, chosen so that each glimpse changes the canvas a lot and the change shows.
For each scene, greedily: from the canvas so far, every viewpoint on a grid of centers and scales is tried, and the
one whose glimpse changes the canvas most is kept; the canvas change of a cell is canvit_pytorch.viz.web's (one minus
the cosine similarity of its layer-normalized patch before and after), averaged over the grid. Writes, per scene, the
viewpoints chosen and their changes, with the runner-up's change at each step, one JSON line per scene as it finishes.

The deck's scenes come from a screen of 120 validation scenes with small glimpses (--sample 120 --scales 0.25 0.35
--step 0.25; build_deck_data.sh): small glimpses leave each later glimpse something to add. The first glimpse always
changes a fresh canvas everywhere, so scenes were ranked by the summed change of glimpses 2 to 5 and the teaser's chosen
by eye among the best, on their canvases."""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from canvit_pytorch import Viewpoint, sample_at_viewpoint
from canvit_pytorch.viz.released_model import CANVAS_GRID_SIZE, GLIMPSE_SIZE_PX, load_released_segmenter
from canvit_pytorch.viz.web import canvas_change

from experiments import logs
from experiments.ade20k import SCENE_PX, load, validation_ids
from experiments.outputs import WORK

log = logging.getLogger(__name__)

BATCH = 32


@dataclass(frozen=True)
class Config:
    scenes: tuple[str, ...] = ()
    """ADE20K validation image ids; with --sample, a random sample of the validation set instead."""
    sample: int = 0
    seed: int = 0
    glimpses: int = 5
    scales: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
    step: float = 0.1
    """Spacing of candidate centers along each axis."""
    out: Path = WORK / "title_teaser/sweep.jsonl"
    device: str = "mps"

    @property
    def candidates(self) -> list[tuple[float, float, float]]:
        return [(float(r), float(c), s) for s in self.scales
                for r in np.arange(-1 + s, 1 - s + 1e-6, self.step) for c in np.arange(-1 + s, 1 - s + 1e-6, self.step)]


@torch.inference_mode()
def greedy(model, scene: torch.Tensor, cfg: Config) -> list[dict]:
    device = scene.device
    canvit = model.canvit
    candidates = cfg.candidates
    state = model.init_state(batch_size=1, canvas_grid_size=CANVAS_GRID_SIZE)
    chosen = []
    for t in range(cfg.glimpses):
        before = canvit.canvas_patches(state.canvas)[0].float().cpu().numpy()
        scores = []
        for i in range(0, len(candidates), BATCH):
            chunk = candidates[i:i + BATCH]
            n = len(chunk)
            viewpoint = Viewpoint(centers=torch.tensor([[r, c] for r, c, _ in chunk], dtype=torch.float32, device=device),
                                  scales=torch.tensor([s for *_, s in chunk], dtype=torch.float32, device=device))
            batch_state = type(state)(**{k: v.expand(n, *v.shape[1:]).clone() for k, v in vars(state).items()})
            glimpse = sample_at_viewpoint(spatial=scene.expand(n, -1, -1, -1), viewpoint=viewpoint, glimpse_size_px=GLIMPSE_SIZE_PX)
            _, after = model(glimpse=glimpse, state=batch_state, viewpoint=viewpoint)
            for patches in canvit.canvas_patches(after.canvas).float().cpu().numpy():
                scores.append(float(canvas_change(before, patches).mean()))
        order = np.argsort(scores)[::-1]
        best = int(order[0])
        r, c, s = candidates[best]
        chosen.append({"t": t, "row": round(r, 3), "col": round(c, 3), "scale": s,
                       "change": scores[best], "runner_up": scores[int(order[1])]})
        viewpoint = Viewpoint(centers=torch.tensor([[r, c]], device=device), scales=torch.tensor([s], device=device))
        _, state = model(glimpse=sample_at_viewpoint(spatial=scene, viewpoint=viewpoint, glimpse_size_px=GLIMPSE_SIZE_PX),
                         state=state, viewpoint=viewpoint)
    return chosen


def main(cfg: Config) -> None:
    assert bool(cfg.scenes) != bool(cfg.sample), "pass --scenes or --sample"
    scenes = list(cfg.scenes) or random.Random(cfg.seed).sample(validation_ids(), cfg.sample)
    device = torch.device(cfg.device)
    model = load_released_segmenter(scene_size_px=SCENE_PX, canvas_grid_size=CANVAS_GRID_SIZE, device=device).model
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    log.info("%d scenes, %d candidates per glimpse, into %s", len(scenes), len(cfg.candidates), cfg.out)
    with cfg.out.open("w") as out:
        for i, image_id in enumerate(scenes):
            image, _ = load(image_id)
            chosen = greedy(model, image.unsqueeze(0).to(device), cfg)
            out.write(json.dumps({"image_id": image_id, "glimpses": chosen}) + "\n")
            out.flush()
            log.info("%d/%d %s: changes %s", i + 1, len(scenes), image_id, " ".join(f"{g['change']:.3f}" for g in chosen))


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
