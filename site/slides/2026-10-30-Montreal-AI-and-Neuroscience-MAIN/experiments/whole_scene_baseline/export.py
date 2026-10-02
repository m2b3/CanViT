"""DINOv3 ViT-B on each recorded rollout's whole scene at a glimpse's resolution (experiments.dinov3_whole_scene), for
#rollout's contrast with CanViT's canvas. Per bundle, into <out>/<bundle name>/:
  labels.png   the decoded class of each DINOv3 patch (grid x grid, uint8, value i is the probe's class i); upsample
               nearest only to display
  meta.json    the image, the models, the grid, and the pixel accuracy measured as the bundle's own
               (canvit_pytorch.viz.web.pixel_accuracy: logits upsampled bilinearly to the annotation)"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from canvit_pytorch.teacher import TEACHER_REPO
from canvit_pytorch.viz.web import cell_center_labels, pixel_accuracy
from PIL import Image

from experiments import logs
from experiments.ade20k import SCENE_PX, load, pixels
from experiments.dinov3_whole_scene import WholeSceneDINOv3
from experiments.glimpses import GLIMPSE_PX
from experiments.outputs import DECK_DATA, SITE

log = logging.getLogger(__name__)

ADE20K_SOURCE_PREFIX = "ADE20K validation set, "


@dataclass(frozen=True)
class Config:
    bundles: tuple[str, ...] = ("street-riid", "shop-riid", "ferry-riid")
    """recorded bundles under site/data/ (record_bundles.sh)"""
    out: Path = DECK_DATA / "whole-scene-dinov3"
    device: str = "mps"


def main(cfg: Config) -> None:
    dinov3 = WholeSceneDINOv3(torch.device(cfg.device))
    for name in cfg.bundles:
        bundle = SITE / "data" / name
        manifest = json.loads((bundle / "manifest.json").read_text())
        source = manifest["scene"]["source"]
        assert source.startswith(ADE20K_SOURCE_PREFIX), f"{bundle}: scene from {source!r}, not ADE20K validation"
        image_id = source.removeprefix(ADE20K_SOURCE_PREFIX)
        image, annotation = load(image_id)
        # The bundle recorded the same scene and annotation: its scene.png and its cell-center truth.
        assert manifest["scene"]["px"] == SCENE_PX, manifest["scene"]
        assert (np.asarray(Image.open(bundle / "scene.png").convert("RGB")) == pixels(image)).all(), bundle
        truth = np.asarray(Image.open(bundle / manifest["scene"]["truth"]))
        assert (truth == cell_center_labels(annotation, manifest["canvas_grid"])).all(), bundle

        logits = dinov3.logits(image)
        out = cfg.out / name
        out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(logits.argmax(0).astype(np.uint8)).save(out / "labels.png")
        accuracy = pixel_accuracy(logits, annotation)
        (out / "meta.json").write_text(json.dumps({
            "image_id": image_id, "bundle": name, "input_px": GLIMPSE_PX, "grid": dinov3.grid, "teacher": TEACHER_REPO,
            "probe": dinov3.probe_repo, "pixel_accuracy": accuracy}, indent=1) + "\n")
        log.info("%s (%s): DINOv3 at %d px, pixel accuracy %.3f; CanViT after the last glimpse %.3f", name, image_id,
                 GLIMPSE_PX, accuracy, manifest["glimpses"][-1]["pixel_accuracy"])


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
