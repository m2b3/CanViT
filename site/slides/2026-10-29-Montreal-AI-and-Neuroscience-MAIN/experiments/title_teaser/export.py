"""Record the title teaser's scenes along the viewpoints sweep.py chose, as web bundles the teaser plays
(data/talk/title-teaser/<image id>), with their ADE20K annotations."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from canvit_pytorch.policies import FixedSequence
from canvit_pytorch.provenance import provenance
from canvit_pytorch.viewpoint import Viewpoint
from canvit_pytorch.viz.record import record
from canvit_pytorch.viz.released_model import CANVAS_GRID_SIZE, GLIMPSE_SIZE_PX, load_released_segmenter
from canvit_pytorch.viz.web import write_bundle

from experiments import logs
from experiments.ade20k import SCENE_PX, load
from experiments.outputs import DECK_DATA, WORK

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    scenes: tuple[str, ...]
    """ADE20K validation image ids, each a line of the sweep."""
    sweep: Path = WORK / "title_teaser/fine.jsonl"
    out: Path = DECK_DATA / "title-teaser"
    device: str = "cpu"


def main(cfg: Config) -> None:
    chosen = {d["image_id"]: d["glimpses"] for d in map(json.loads, cfg.sweep.read_text().splitlines())}
    missing = [s for s in cfg.scenes if s not in chosen]
    assert not missing, f"{cfg.sweep} has no viewpoints for {missing}"
    device = torch.device(cfg.device)
    released = load_released_segmenter(scene_size_px=SCENE_PX, canvas_grid_size=CANVAS_GRID_SIZE, device=device)
    for image_id in cfg.scenes:
        image, annotation = load(image_id)
        glimpses = chosen[image_id]
        policy = FixedSequence([Viewpoint(centers=torch.tensor([[g["row"], g["col"]]], device=device),
                                          scales=torch.tensor([g["scale"]], device=device)) for g in glimpses])
        rollout = record(released.model, image.unsqueeze(0).to(device), policy, num_glimpses=len(glimpses),
                         canvas_grid_size=CANVAS_GRID_SIZE, glimpse_size_px=GLIMPSE_SIZE_PX, capture_writes=False,
                         annotation=torch.from_numpy(annotation))
        path = write_bundle(
            rollout, cfg.out / image_id, title=f"ADE20K {image_id}",
            scene={"source": f"ADE20K validation set, {image_id}", "attribution": "ADE20K (MIT CSAIL)",
                   "license": "ADE20K terms of use", "annotation_source": f"{image_id}.png"},
            model=released.model_record, readout=released.readout_record,
            policy={"name": "largest_canvas_change", "paper_name": None,
                    "description": "each viewpoint the one, on a grid, whose glimpse changes the canvas most "
                                   "(experiments/title_teaser/sweep.py)",
                    "deterministic": True, "seed": None},
            pca_protocol="fixed-limits",
            provenance={**provenance(device), "precision": "float32"},
        )
        log.info("wrote %s", path)


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
