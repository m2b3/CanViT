"""DINOv3 ViT-B on the whole scene at 128 px, the same input budget as CanViT's full-scene glimpse, decoded by its
released 128 px ADE20K probe: its probability of a plotted object's class per patch (8 x 8), drawn as plot.py draws
CanViT's (inferno, 0 to 1, nearest), and its mean over the object's pixels, the measure of meta.json's prob_f. Writes
<dir>/<export>/prob_dinov3_128.png and dinov3_128.json beside plot.py's panels."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from canvit_pytorch.hub.repos import released_dinov3_ade20k_probe
from canvit_pytorch.probes import SegmentationProbe
from canvit_pytorch.teacher import TEACHER_REPO, load_teacher
from canvit_pytorch.viewpoint import Viewpoint, sample_at_viewpoint
from PIL import Image

from experiments import logs
from experiments.ade20k import SCENE_PX, load
from experiments.glimpses import GLIMPSE_PX
from experiments.looking_closer.definition import small_objects

log = logging.getLogger(__name__)

INPUT_PX = GLIMPSE_PX


@dataclass(frozen=True)
class Config:
    separate: Path
    """the directory plot.py wrote its panels into"""
    export: str
    """<image id>-<class name>, one of its subdirectories"""
    device: str = "mps"


@torch.inference_mode()
def main(cfg: Config) -> None:
    out = cfg.separate / cfg.export
    meta = json.loads((out / "meta.json").read_text())
    image, labels = load(meta["image_id"])
    (obj, mask), = [(o, m) for o, m in small_objects(meta["image_id"], labels) if o.name == meta["class"]]
    device = torch.device(cfg.device)
    teacher = load_teacher(TEACHER_REPO, device)
    probe_repo = released_dinov3_ade20k_probe("dv3b", input_size_px=INPUT_PX)
    probe = SegmentationProbe.from_pretrained(probe_repo).to(device).eval()
    crop = sample_at_viewpoint(spatial=image[None].to(device), viewpoint=Viewpoint.full_scene(batch_size=1, device=device),
                               glimpse_size_px=INPUT_PX)
    grid = INPUT_PX // teacher.patch_size
    logits = probe(teacher(crop).patches.view(1, grid, grid, -1).float())[0].cpu()  # [C, grid, grid]
    p = logits.softmax(0)[obj.cls]
    on_scene = F.interpolate(p[None, None], size=(SCENE_PX, SCENE_PX), mode="nearest-exact")[0, 0].numpy()
    prob = float(on_scene[mask].mean())
    colors = (matplotlib.colormaps["inferno"](on_scene)[..., :3] * 255).round().astype(np.uint8)
    Image.fromarray(colors).save(out / "prob_dinov3_128.png")
    (out / "dinov3_128.json").write_text(json.dumps({
        "image_id": meta["image_id"], "class": meta["class"], "input_px": INPUT_PX, "teacher": TEACHER_REPO,
        "probe": probe_repo, "prob": prob}, indent=1))
    log.info("%s: DINOv3 at %d px, mean p(%s) over the object %.3f (CanViT full-scene glimpse %.3f, after the zoom "
             "%.3f)", out, INPUT_PX, meta["class"], prob, meta["prob_f"], meta["prob_fz"])


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
