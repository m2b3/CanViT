"""Every large object of ADE20K validation (classes in TARGETS): glimpse its two ends, never its middle, and measure
how much of the unseen middle CanViT's canvas labels as the object after A, after B and after A then B. Writes the
examples with their measures (metrics only) and logs the means; export.py saves the data of chosen examples."""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F
import tyro
from canvit_pytorch.benchmarks.ade20k import CLASS_NAMES, IGNORE_LABEL
from canvit_pytorch.viz.released_model import CANVAS_GRID_SIZE, load_released_segmenter

from experiments import logs
from experiments.ade20k import SCENE_PX, load, validation_ids
from experiments.outputs import WORK
from experiments.table_corners.definition import Example, canvit_logits, regions

log = logging.getLogger(__name__)

TARGETS = ["table", "desk", "coffee table", "pool table", "bed", "sofa", "car", "bus", "truck", "van", "boat", "ship",
           "airplane", "counter", "countertop", "kitchen island", "bench", "bookcase", "wardrobe", "bathtub", "cabinet",
           "chest of drawers", "bar", "fireplace", "bridge", "fence", "railing", "building", "house"]
TARGET_IDS = {CLASS_NAMES.index(name): name for name in TARGETS}
MIN_AREA, MAX_AREA = 0.04, 0.6  # of the scene, for the object's largest connected component
END_QUANTILES = (0.12, 0.88)  # where the glimpses go along the object's longer axis
GLIMPSE_SIDE = (0.28, 48, 128)  # fraction of the object's extent, clamped to [min, max] px
MIN_MIDDLE_AREA = 0.015  # of the scene: the unseen middle must be this large to score
BATCH = 16


@dataclass(frozen=True)
class Config:
    out: Path = WORK / "table_corners/examples.json"
    device: str = "mps"


def candidates(image_id: str, labels: np.ndarray) -> list[Example]:
    found = []
    for cls, name in TARGET_IDS.items():
        mask = labels == cls
        if mask.mean() < MIN_AREA:
            continue
        components, count = ndi.label(mask)
        sizes = ndi.sum(mask, components, range(1, count + 1))
        component = components == (1 + int(np.argmax(sizes)))
        area = float(component.mean())
        if not MIN_AREA <= area <= MAX_AREA:
            continue
        rows, cols = np.nonzero(component)
        height, width = np.ptp(rows) + 1, np.ptp(cols) + 1
        horizontal = bool(width >= height)
        along, across = (cols, rows) if horizontal else (rows, cols)
        fraction, low, high = GLIMPSE_SIDE
        s = float(np.clip(fraction * (width if horizontal else height), low, high)) / SCENE_PX
        ends = []
        for q in END_QUANTILES:
            pos = np.quantile(along, q)
            cross = float(np.median(across[np.abs(along - pos) <= max(4, 0.02 * SCENE_PX)]))
            r, c = (cross, pos) if horizontal else (pos, cross)
            ends.append((float(np.clip(r / (SCENE_PX / 2) - 1, s - 1, 1 - s)),
                         float(np.clip(c / (SCENE_PX / 2) - 1, s - 1, 1 - s)), s))
        ex = Example(image_id, cls, name, horizontal, ends[0], ends[1], area, 0.0)
        ex.middle_area = float(regions(ex, labels)[0].mean())
        if ex.middle_area >= MIN_MIDDLE_AREA:
            found.append(ex)
    return found


def labels_at_scene(logits: torch.Tensor) -> np.ndarray:
    """[B, C, g, g] -> [B, S, S]: bilinear upsampling, then argmax, as the paper's evaluation does."""
    return torch.cat([F.interpolate(c, size=(SCENE_PX, SCENE_PX), mode="bilinear", align_corners=False).argmax(1).cpu()
                      for c in logits.split(4)]).numpy()


@torch.inference_mode()
def main(cfg: Config) -> None:
    start = time.time()
    examples = [ex for image_id in validation_ids() for ex in candidates(image_id, load(image_id)[1])]
    log.info("%d candidates in %.0fs", len(examples), time.time() - start)
    device = torch.device(cfg.device)
    model = load_released_segmenter(scene_size_px=SCENE_PX, canvas_grid_size=CANVAS_GRID_SIZE, device=device).model
    for i in range(0, len(examples), BATCH):
        chunk = examples[i:i + BATCH]
        loaded = [load(ex.image_id) for ex in chunk]
        images = torch.stack([image for image, _ in loaded]).to(device)
        pa = labels_at_scene(canvit_logits(model, images, [[ex.a for ex in chunk]]))
        pb = labels_at_scene(canvit_logits(model, images, [[ex.b for ex in chunk]]))
        pab = labels_at_scene(canvit_logits(model, images, [[ex.a for ex in chunk], [ex.b for ex in chunk]]))
        for j, (ex, (_, labels)) in enumerate(zip(chunk, loaded)):
            middle, corridor = regions(ex, labels)
            ex.mid_a, ex.mid_b, ex.mid_ab = (float((p[j][middle] == ex.cls).mean()) for p in (pa, pb, pab))
            other = corridor & (labels != ex.cls) & (labels != IGNORE_LABEL)
            ex.fp_ab = float((pab[j][other] == ex.cls).mean()) if other.any() else 0.0
        if (i // BATCH) % 10 == 0:
            log.info("inference %d/%d, %.0fs", i, len(examples), time.time() - start)
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    cfg.out.write_text(json.dumps([asdict(ex) for ex in examples], indent=1))
    mids = np.array([[ex.mid_a, ex.mid_b, ex.mid_ab] for ex in examples])
    log.info("%d examples; mean middle recall: A %.3f  B %.3f  AB %.3f; mean fp AB %.3f -> %s", len(examples),
             *mids.mean(0), np.mean([ex.fp_ab for ex in examples]), cfg.out)


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
