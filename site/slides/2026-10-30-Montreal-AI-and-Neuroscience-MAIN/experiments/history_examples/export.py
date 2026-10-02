"""Save, uncolored, chosen rows of the ranked sweeps (rank.py) for the #history slide, into <out>/:
  classification/<photo>/  image_512.png   what the probe saw (canvit_pytorch.preprocess.preprocess(512))
                           image.png       the photo, aspect kept, longest side at most MAX_SIDE_PX
                           label.json      its class, the probe's probability of it and the runner-up
  segmentation/<ADE id>/   image_512.png, labels_512.png   the evaluation transform's scene, squashed to 512 x 512
                           image.png, labels.png           aspect kept, longest side at most MAX_SIDE_PX
                           classes.json    the pixel fraction of each class present
Label maps are uint8: value i is CLASS_NAMES[i] (the probes' output order), 255 unlabeled. plot.py colors a
segmentation example for the slide."""

import csv
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from canvit_pytorch.benchmarks.ade20k import (
    CLASS_NAMES,
    IGNORE_LABEL,
    dataset_root,
    decode_annotation,
    evaluation_transform,
)
from canvit_pytorch.hub.repos import DINOV3_VITB16_IN1K_PROBE
from canvit_pytorch.preprocess import imagenet_denormalize, preprocess
from canvit_pytorch.teacher import TEACHER_REPO
from PIL import Image

from experiments import logs
from experiments.ade20k import SCENE_PX
from experiments.outputs import DECK_DATA, WORK

log = logging.getLogger(__name__)

MAX_SIDE_PX = 1024


@dataclass(frozen=True)
class Config:
    imagenette: Path
    """Imagenette 2's val/ directory, as given to classify_sweep.py"""
    classification: tuple[str, ...]
    """photos by file name without extension (ILSVRC2012_val_00037182)"""
    segmentation: tuple[str, ...]
    """ADE20K validation ids (ADE_val_00001509)"""
    work: Path = WORK / "history"
    """where rank.py wrote classification/ranked.tsv and segmentation/ranked.tsv"""
    out: Path = DECK_DATA / "history"


def read_ranked(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def to_pil(normalized: torch.Tensor) -> Image.Image:
    """An ImageNet-normalized [3, H, W] image as the uint8 RGB image it was made from."""
    pixels = (imagenet_denormalize(normalized) * 255).round().to(torch.uint8)
    return Image.fromarray(pixels.permute(1, 2, 0).numpy())


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


def export_classification(row: dict[str, str], imagenette: Path, out: Path) -> None:
    assert row["top1_index"] == row["class_index"], f"{row['image']}: the probe gets it wrong"
    photo = Image.open(imagenette / row["image"]).convert("RGB")
    out = fresh_dir(out / Path(row["image"]).stem)
    to_pil(preprocess(SCENE_PX)(photo)).save(out / "image_512.png")
    photo.resize(export_size(photo.size), Image.Resampling.LANCZOS).save(out / "image.png")
    record = {
        "_about": "An ImageNet-1k photo and the passive classifier's probability of its true class. probability: "
                  "softmax of DINOv3 ViT-B/16's CLS token (after its final LayerNorm) through the linear ImageNet-1k "
                  "probe, on image_512.png (canvit_pytorch.preprocess.preprocess(512): short side resized to 512 px, "
                  "bilinear, then center-cropped). imagenet_split: 'val' photos are held out from the probe's "
                  "training, 'train' photos are in it.",
        "source": f"imagenette2/val/{row['image']}", "imagenet_split": row["imagenet_split"],
        "original_size": [int(row["width"]), int(row["height"])],
        "wnid": row["wnid"], "class_index": int(row["class_index"]), "class_name": row["class_name"],
        "probability": float(row["prob_true"]),
        "runner_up": {"class_name": row["top2_name"], "probability": float(row["top2_prob"])},
        "backbone": TEACHER_REPO, "probe": DINOV3_VITB16_IN1K_PROBE,
    }
    (out / "label.json").write_text(json.dumps(record, indent=2) + "\n")
    log.info("exported %s", out)


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

    image_512, labels_512 = evaluation_transform(SCENE_PX)(image, annotation)
    to_pil(image_512).save(out / "image_512.png")
    labels_to_pil(labels_512).save(out / "labels_512.png")

    size = export_size(image.size)
    image.resize(size, Image.Resampling.LANCZOS).save(out / "image.png")
    labels = decode_annotation(torch.from_numpy(np.array(annotation.resize(size, Image.Resampling.NEAREST))))
    labels_to_pil(labels).save(out / "labels.png")

    classes, unlabeled = class_fractions(decode_annotation(torch.from_numpy(np.array(annotation))))
    classes_512, unlabeled_512 = class_fractions(labels_512)
    record = {
        "_about": "An ADE20K validation scene and its annotation. labels*.png: uint8, value i is class i "
                  "(canvit_pytorch.benchmarks.ade20k.CLASS_NAMES, the probes' output order), 255 unlabeled. "
                  "classes: pixel fraction of each class present in the original annotation; classes_512: in "
                  "labels_512.png.",
        "id": row["id"], "scene": row["scene"], "original_size": list(image.size), "image_size": list(size),
        "unlabeled_index": IGNORE_LABEL, "unlabeled_fraction": unlabeled, "classes": classes,
        "unlabeled_fraction_512": unlabeled_512, "classes_512": classes_512,
    }
    (out / "classes.json").write_text(json.dumps(record, indent=2) + "\n")
    log.info("exported %s", out)


def rows_named(names: tuple[str, ...], rows: list[dict[str, str]], column: str) -> list[dict[str, str]]:
    """The rows whose `column`, without directory or extension, is each of `names`."""
    by_name = {Path(row[column]).stem: row for row in rows}
    missing = [name for name in names if name not in by_name]
    assert not missing, f"not in the ranked sweep: {missing}"
    return [by_name[name] for name in names]


def main(cfg: Config) -> None:
    for row in rows_named(cfg.classification, read_ranked(cfg.work / "classification/ranked.tsv"), "image"):
        export_classification(row, cfg.imagenette, cfg.out / "classification")
    for row in rows_named(cfg.segmentation, read_ranked(cfg.work / "segmentation/ranked.tsv"), "id"):
        export_segmentation(row, cfg.out / "segmentation")


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
