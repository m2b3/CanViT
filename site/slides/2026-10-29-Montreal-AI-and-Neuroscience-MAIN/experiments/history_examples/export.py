"""Save, uncolored, chosen rows of the ranked segmentation sweep (rank.py) for #gap, into <out>/<ADE id>/: image.png
and labels.png (aspect kept, longest side at most MAX_SIDE_PX) and classes.json (the pixel fraction of each class
present). labels.png is uint8: value i is CLASS_NAMES[i] (the probes' output order), 255 unlabeled. plot.py colors it
for the slide."""

import csv
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from canvit_pytorch.benchmarks.ade20k import CLASS_NAMES, IGNORE_LABEL, dataset_root, decode_annotation
from PIL import Image

from experiments import logs
from experiments.outputs import DECK_DATA, WORK

log = logging.getLogger(__name__)

MAX_SIDE_PX = 1024


@dataclass(frozen=True)
class Config:
    segmentation: tuple[str, ...]
    """ADE20K validation ids (ADE_val_00001509)"""
    work: Path = WORK / "history"
    """where rank.py wrote segmentation/ranked.tsv"""
    out: Path = DECK_DATA / "history/segmentation"


def read_ranked(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def labels_to_pil(labels: torch.Tensor) -> Image.Image:
    assert labels.ndim == 2 and int(labels.min()) >= 0 and int(labels.max()) <= IGNORE_LABEL
    return Image.fromarray(labels.to(torch.uint8).numpy(), mode="L")


def export_size(size: tuple[int, int]) -> tuple[int, int]:
    """Aspect kept, longest side at most MAX_SIDE_PX; never enlarged."""
    scale = min(1.0, MAX_SIDE_PX / max(size))
    return round(size[0] * scale), round(size[1] * scale)


def fresh_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def class_fractions(labels: torch.Tensor) -> tuple[dict[str, dict[str, object]], float]:
    counts = torch.bincount(labels.flatten(), minlength=IGNORE_LABEL + 1)
    total = labels.numel()
    present = sorted((i for i in range(len(CLASS_NAMES)) if counts[i] > 0), key=lambda i: -int(counts[i]))
    classes = {str(i): {"name": CLASS_NAMES[i], "fraction": round(int(counts[i]) / total, 6)} for i in present}
    return classes, round(int(counts[IGNORE_LABEL]) / total, 6)


def export_segmentation(row: dict[str, str], out: Path) -> None:
    assert row["failed"] == "-" and row["excluded"] == "0", f"{row['id']} fails the slide's criteria: {row['failed']}"
    root = dataset_root()
    image = Image.open(root / "images/validation" / f"{row['id']}.jpg").convert("RGB")
    annotation = Image.open(root / "annotations/validation" / f"{row['id']}.png")
    assert image.size == annotation.size, (image.size, annotation.size)
    out = fresh_dir(out / row["id"])
    size = export_size(image.size)
    image.resize(size, Image.Resampling.LANCZOS).save(out / "image.png")
    labels = decode_annotation(torch.from_numpy(np.array(annotation.resize(size, Image.Resampling.NEAREST))))
    labels_to_pil(labels).save(out / "labels.png")

    classes, unlabeled = class_fractions(decode_annotation(torch.from_numpy(np.array(annotation))))
    record = {
        "_about": "An ADE20K validation scene and its annotation. labels.png: uint8, value i is class i "
                  "(canvit_pytorch.benchmarks.ade20k.CLASS_NAMES, the probes' output order), 255 unlabeled. "
                  "classes: pixel fraction of each class present in the original annotation.",
        "id": row["id"], "scene": row["scene"], "original_size": list(image.size), "image_size": list(size),
        "unlabeled_index": IGNORE_LABEL, "unlabeled_fraction": unlabeled, "classes": classes,
    }
    (out / "classes.json").write_text(json.dumps(record, indent=2) + "\n")
    log.info("exported %s", out)


def rows_named(names: tuple[str, ...], rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """The rows whose id is each of `names`."""
    by_name = {row["id"]: row for row in rows}
    missing = [name for name in names if name not in by_name]
    assert not missing, f"not in the ranked sweep: {missing}"
    return [by_name[name] for name in names]


def main(cfg: Config) -> None:
    for row in rows_named(cfg.segmentation, read_ranked(cfg.work / "segmentation/ranked.tsv")):
        export_segmentation(row, cfg.out)


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
