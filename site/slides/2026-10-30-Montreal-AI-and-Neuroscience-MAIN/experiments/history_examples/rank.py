"""Rank the sweeps' rows by the properties OUTLINE.md asks of each example (#history, "Examples").

classification: ImageNet-1k validation photos first (held out from the probe's training; added to the requested
criteria), then correct top-1, then short side >= MIN_SHORT_SIDE_PX, then the probability of the true class.

segmentation: by the number of criteria failed, then the number of classes whose region holds a disk of radius
LABEL_RADIUS_FRACTION of the short side (room for the class name), then the unlabeled fraction. The excluded image goes
last.
"""

import csv
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import tyro

from experiments import logs
from experiments.outputs import WORK

log = logging.getLogger(__name__)

# Classification
MIN_SHORT_SIDE_PX = 375

# Segmentation
CLASS_FRACTION = 0.005
MIN_CLASSES, MAX_CLASSES = 7, 14  # classes covering >= CLASS_FRACTION
SMALL_FRACTION_RANGE = (0.001, 0.02)
MIN_SMALL_CLASSES = 3
MAX_UNLABELED = 0.03
MAX_LARGEST_CLASS = 0.45
PREFERRED_SCENES = frozenset({"living_room", "bedroom", "kitchen", "street", "dining_room", "office", "bathroom"})
MIN_SCENE_SHORT_SIDE_PX = 480  # 326 validation images are 256 x 256, too coarse to show large
LABEL_RADIUS_FRACTION = 0.04
EXCLUDED = frozenset({"ADE_val_00001271"})  # the conference room, shown under other claims


@dataclass(frozen=True)
class Config:
    work: Path = WORK / "history"
    """where the sweeps wrote classification/scores.tsv and segmentation/stats.tsv; the rankings go beside them"""


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    log.info("wrote %s (%d rows)", path, len(rows))


def rank_classification(directory: Path) -> None:
    ranked = []
    for row in read_tsv(directory / "scores.tsv"):
        ranked.append({
            "held_out": int(row["imagenet_split"] == "val"),
            "correct": int(row["top1_index"] == row["class_index"]),
            "short_side_ok": int(int(row["short_side"]) >= MIN_SHORT_SIDE_PX),
        } | row)
    ranked.sort(key=lambda r: (-r["held_out"], -r["correct"], -r["short_side_ok"], -float(r["prob_true"])))
    ranked = [{"rank": rank} | row for rank, row in enumerate(ranked, start=1)]
    passing = [r for r in ranked if r["held_out"] and r["correct"] and r["short_side_ok"]]
    log.info("classification: %d photos, %d held out, %d held out + correct + short side >= %d px; per class: %s",
             len(ranked), sum(r["held_out"] for r in ranked), len(passing), MIN_SHORT_SIDE_PX,
             dict(Counter(r["class_name"] for r in passing)))
    write_tsv(directory / "ranked.tsv", ranked)


def parse_fractions(field: str) -> dict[str, float]:
    return {name: float(value) for name, value in (item.rsplit("=", 1) for item in field.split(";") if item)}


def rank_segmentation(directory: Path) -> None:
    ranked = []
    for row in read_tsv(directory / "stats.tsv"):
        fractions = parse_fractions(row["classes"])
        radii = parse_fractions(row["inscribed_radius"])
        counted = [name for name, fraction in fractions.items() if fraction >= CLASS_FRACTION]
        small = [name for name, f in fractions.items() if SMALL_FRACTION_RANGE[0] <= f <= SMALL_FRACTION_RANGE[1]]
        labelable = [name for name, radius in radii.items() if radius >= LABEL_RADIUS_FRACTION]
        largest_name, largest = max(fractions.items(), key=lambda item: item[1])
        unlabeled = float(row["unlabeled"])
        short_side = min(int(row["width"]), int(row["height"]))
        failed = [name for name, ok in {
            "class_count": MIN_CLASSES <= len(counted) <= MAX_CLASSES,
            "small": len(small) >= MIN_SMALL_CLASSES,
            "unlabeled": unlabeled < MAX_UNLABELED,
            "largest": largest <= MAX_LARGEST_CLASS,
            "scene": row["scene"] in PREFERRED_SCENES,
            "resolution": short_side >= MIN_SCENE_SHORT_SIDE_PX,
        }.items() if not ok]
        ranked.append({
            "id": row["id"], "scene": row["scene"], "width": row["width"], "height": row["height"],
            "excluded": int(row["id"] in EXCLUDED), "failed": ",".join(failed) or "-",
            "n_classes": len(counted), "n_small": len(small), "n_labelable": len(labelable),
            "unlabeled": f"{unlabeled:.4f}", "largest": f"{largest:.4f}", "largest_class": largest_name,
            "labelable": ", ".join(labelable), "small": ", ".join(small),
            "classes": ", ".join(f"{name} {100 * f:.1f}%" for name, f in fractions.items() if f >= CLASS_FRACTION),
        })
    ranked.sort(key=lambda r: (r["excluded"], len(r["failed"].split(",")) if r["failed"] != "-" else 0,
                               -r["n_labelable"], float(r["unlabeled"])))
    ranked = [{"rank": rank} | row for rank, row in enumerate(ranked, start=1)]
    passing = [r for r in ranked if r["failed"] == "-" and not r["excluded"]]
    log.info("segmentation: %d images, %d pass every criterion; by scene: %s; failures: %s",
             len(ranked), len(passing), dict(Counter(r["scene"] for r in passing)),
             dict(Counter(name for r in ranked for name in r["failed"].split(",") if name != "-")))
    write_tsv(directory / "ranked.tsv", ranked)


def main(cfg: Config) -> None:
    rank_segmentation(cfg.work / "segmentation")
    rank_classification(cfg.work / "classification")


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
