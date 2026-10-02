"""The #history slide's segmentation example drawn from export.py's data: its annotation as a colored map, and where
each class name goes. Reads <data>/<scene>/{labels.png, classes.json}; writes beside them map.png (a color per class,
by decreasing area; white where unlabeled) and names.json (each class covering >= MIN_NAMED_FRACTION: its name, the
center of the largest disk inside its region as fractions of the image's width and height, that disk's radius in
pixels)."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from canvit_pytorch.benchmarks.ade20k import IGNORE_LABEL
from PIL import Image
from scipy.ndimage import distance_transform_edt

from experiments.outputs import DECK_DATA

MIN_NAMED_FRACTION = 0.005
# Muted, distinguishable fills (Tableau 20's lighter half, then its darker half).
PALETTE = ["#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#dbdb8d", "#9edae5",
           "#c7c7c7", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22",
           "#17becf", "#7f7f7f"]


@dataclass(frozen=True)
class Config:
    scene: str
    """an ADE20K validation id that export.py wrote"""
    data: Path = DECK_DATA / "history/segmentation"


def rgb(hex_color: str) -> tuple[int, int, int]:
    return int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)


def main(cfg: Config) -> None:
    folder = cfg.data / cfg.scene
    record = json.loads((folder / "classes.json").read_text())
    labels = np.array(Image.open(folder / "labels.png"))
    assert labels.ndim == 2, labels.shape
    by_area = sorted(record["classes"].items(), key=lambda item: -item[1]["fraction"])
    assert len(by_area) <= len(PALETTE), f"{len(by_area)} classes, {len(PALETTE)} colors"
    colored = np.full((*labels.shape, 3), 255, dtype=np.uint8)
    assert not np.isin(labels, [int(index) for index, _ in by_area] + [IGNORE_LABEL], invert=True).any()
    names = []
    for (index, info), color in zip(by_area, PALETTE, strict=False):
        region = labels == int(index)
        assert region.any(), f"class {index} ({info['name']}) is listed but absent from labels.png"
        colored[region] = rgb(color)
        if info["fraction"] < MIN_NAMED_FRACTION:
            continue
        distance = distance_transform_edt(np.pad(region, 1))[1:-1, 1:-1]
        y, x = np.unravel_index(distance.argmax(), distance.shape)
        names.append({"name": info["name"], "x": round((x + 0.5) / labels.shape[1], 4),
                      "y": round((y + 0.5) / labels.shape[0], 4), "radius_px": round(float(distance.max()), 1),
                      "fraction": info["fraction"]})
    Image.fromarray(colored).save(folder / "map.png")
    (folder / "names.json").write_text(json.dumps({"scene": cfg.scene, "size": list(labels.shape[::-1]),
                                                   "names": names}, indent=1))
    print(f"wrote {folder / 'map.png'} and names.json: {', '.join(n['name'] for n in names)}")


if __name__ == "__main__":
    main(tyro.cli(Config))
