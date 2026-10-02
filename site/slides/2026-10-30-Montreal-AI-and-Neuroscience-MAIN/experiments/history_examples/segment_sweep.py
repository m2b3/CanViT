"""Statistics of every ADE20K validation annotation, for choosing #history's segmentation example. Per image, from its
annotation at full resolution (value 0 unlabeled, k the class CLASS_NAMES[k - 1]): pixel fraction of each class
present, and for each class covering >= MIN_CLASS_FRACTION the radius of the largest disk inside its region (where a
class name could be written), as a fraction of the short side. Writes a table, one row per image, unranked; rank.py
turns the columns into criteria."""

import csv
import logging
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tyro
from canvit_pytorch.benchmarks.ade20k import CLASS_NAMES, NUM_CLASSES, NUM_VALIDATION_IMAGES, dataset_root
from PIL import Image
from scipy.ndimage import distance_transform_edt

from experiments import logs
from experiments.ade20k import scene_categories
from experiments.outputs import WORK

log = logging.getLogger(__name__)

MIN_CLASS_FRACTION = 0.005  # a class "counts" for the example at >= 0.5% of the pixels
NUM_PROCESSES = 8
COLUMNS = ["id", "scene", "width", "height", "unlabeled", "classes", "inscribed_radius"]


@dataclass(frozen=True)
class Config:
    out: Path = WORK / "history/segmentation/stats.tsv"


def image_stats(annotation_path: Path) -> dict[str, str]:
    annotation = np.asarray(Image.open(annotation_path))
    assert annotation.ndim == 2 and annotation.dtype == np.uint8 and annotation.max() <= NUM_CLASSES, annotation_path
    height, width = annotation.shape
    fractions = np.bincount(annotation.ravel(), minlength=NUM_CLASSES + 1) / annotation.size
    present = sorted((k for k in range(1, NUM_CLASSES + 1) if fractions[k] > 0), key=lambda k: -fractions[k])
    radii = {
        k: float(distance_transform_edt(np.pad(annotation == k, 1)).max()) / min(width, height)
        for k in present if fractions[k] >= MIN_CLASS_FRACTION
    }
    return {
        "id": annotation_path.stem, "width": str(width), "height": str(height),
        "unlabeled": f"{fractions[0]:.5f}",
        "classes": ";".join(f"{CLASS_NAMES[k - 1]}={fractions[k]:.5f}" for k in present),
        "inscribed_radius": ";".join(f"{CLASS_NAMES[k - 1]}={radius:.4f}" for k, radius in radii.items()),
    }


def main(cfg: Config) -> None:
    annotations = sorted((dataset_root() / "annotations" / "validation").glob("*.png"))
    assert len(annotations) == NUM_VALIDATION_IMAGES, len(annotations)
    scenes = scene_categories()
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    with Pool(NUM_PROCESSES) as pool, cfg.out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        for done, stats in enumerate(pool.imap(image_stats, annotations, chunksize=8), start=1):
            writer.writerow(stats | {"scene": scenes[stats["id"]]})
            if done % 200 == 0:
                log.info("%d/%d annotations", done, len(annotations))
    log.info("wrote %s", cfg.out)


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
