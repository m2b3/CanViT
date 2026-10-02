"""Pretraining rollouts for #policy-agnosticism, drawn by the package's own pretraining sampler: for each scene, a
rollout length (canvit_pytorch.pretrain.step.sample_rollout_length with the pretraining config's chunk size and stop
probability), then that many viewpoints under each pretraining policy (config.rollout_policies: F-IID and R-IID, via
canvit_pytorch.policies.make_policy). Scenes are ImageNet photographs (Imagenette's ImageNet-1k validation images;
pretraining used ImageNet-21k), center-cropped to 512 px as pretraining's preprocessing does. Writes
<out>/{<name>.png, rollouts.json}."""

import json
import logging
import random
from dataclasses import dataclass, fields
from pathlib import Path

import torch
import tyro
from canvit_pytorch.policies import POLICIES, make_policy
from canvit_pytorch.preprocess import preprocess
from canvit_pytorch.pretrain.config import PretrainingConfig
from canvit_pytorch.pretrain.step import sample_rollout_length
from PIL import Image

from experiments import logs
from experiments.ade20k import pixels
from experiments.outputs import DECK_DATA

log = logging.getLogger(__name__)

SCENES = {  # name: wnid folder (its first validation photograph), photographs anyone reads at a glance
    "dog": "n02102040", "church": "n03028079", "truck": "n03417042", "parachute": "n03888257",
}


@dataclass(frozen=True)
class Config:
    imagenette: Path
    """Imagenette 2's val/ directory (imagenette2/val), one folder per WordNet id"""
    out: Path = DECK_DATA / "pretraining"


def main(cfg: Config) -> None:
    defaults = {f.name: f.default for f in fields(PretrainingConfig)}  # the declared defaults: the paper's recipe
    chunk, stop, policies = defaults["tbptt_chunk_glimpses"], defaults["stop_probability"], defaults["rollout_policies"]
    cfg.out.mkdir(parents=True, exist_ok=True)
    # The first seed whose rollout lengths take at least three different values, so the slide shows that lengths
    # vary; the draws themselves are the sampler's.
    seed = 0
    while True:
        random.seed(seed)
        if len({sample_rollout_length(tbptt_chunk_glimpses=chunk, stop_probability=stop) for _ in SCENES}) >= 3:
            break
        seed += 1
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    scenes = []
    for name, wnid in SCENES.items():
        path = sorted((cfg.imagenette / wnid).glob("ILSVRC2012_val_*.JPEG"))[0]
        Image.fromarray(pixels(preprocess(512)(Image.open(path).convert("RGB")))).save(cfg.out / f"{name}.png")
        length = sample_rollout_length(tbptt_chunk_glimpses=chunk, stop_probability=stop)
        rollouts = {}
        for policy in policies:
            sampler = make_policy(policy, batch_size=1, device=device, num_glimpses=length, canvas_grid_size=32)
            viewpoints = [sampler.step(t, None) for t in range(length)]
            rollouts[policy] = [[float(v.centers[0, 0]), float(v.centers[0, 1]), float(v.scales[0])] for v in viewpoints]
        scenes.append({"name": name, "image": f"{name}.png", "source": str(path.relative_to(cfg.imagenette.parent)),
                       "rollouts": rollouts})
        log.info("%s %s length %d", name, path.name, length)
    (cfg.out / "rollouts.json").write_text(json.dumps({
        "_about": "Viewpoints (row, col, scale; scale = half side in [-1, 1] scene coordinates) drawn by the pretraining "
                  "sampler for these scenes, with the pretraining config's rollout length process; policies keyed by "
                  "canvit_pytorch.policies names.",
        "tbptt_chunk_glimpses": chunk, "stop_probability": stop,
        "policies": {p: POLICIES[p].paper_name for p in policies}, "seed": seed, "scenes": scenes,
    }, indent=1))
    log.info("wrote %s", cfg.out)


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
