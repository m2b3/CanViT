"""Draw the quickstart export (export.py): the scene with each glimpse's box (scene-<n>.png), the labels after each model
call colored with one color per class across calls (map-<n>.png, 64 x 64, for nearest-neighbor display) and where each
class name goes (names-<n>.json, as labeled-map.js reads it: the center of the largest disk inside the class's
region)."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

from experiments import logs
from experiments.outputs import DECK_DATA, SITE

log = logging.getLogger(__name__)

MIN_NAMED_CELLS = 24
GLIMPSE_BLUE = "#2d6cdf"
# Muted, distinguishable fills (Tableau 20's lighter half, then its darker half), assigned by decreasing area.
PALETTE = ["#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#dbdb8d", "#9edae5",
           "#c7c7c7", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22",
           "#17becf", "#7f7f7f"]


@dataclass(frozen=True)
class Config:
    out: Path = DECK_DATA / "quickstart"
    """the directory export.py wrote"""


def main(cfg: Config) -> None:
    record = json.loads((cfg.out / "export.json").read_text())
    names = record["class_names"]
    labels = [np.array(Image.open(cfg.out / f"labels-{n}.png")) for n in range(1, len(record["viewpoints_row_col_scale"]) + 1)]
    present = np.unique(np.concatenate([lab.ravel() for lab in labels]))
    area = {k: sum(int((lab == k).sum()) for lab in labels) for k in present}
    order = sorted(present, key=lambda k: -area[k])
    assert len(order) <= len(PALETTE), f"{len(order)} classes, {len(PALETTE)} colors"
    color = {k: tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for k, c in zip(order, PALETTE)}
    scene = Image.open(SITE.parent / record["scene"]).convert("RGB")
    side = scene.size[0]
    for n, (cells, (row, col, s)) in enumerate(zip(labels, record["viewpoints_row_col_scale"]), start=1):
        colored = np.zeros((*cells.shape, 3), dtype=np.uint8)
        spots = []
        for k in order:
            region = cells == k
            colored[region] = color[k]
            if region.sum() >= MIN_NAMED_CELLS:
                distance = distance_transform_edt(np.pad(region, 1))[1:-1, 1:-1]
                y, x = np.unravel_index(distance.argmax(), distance.shape)
                spots.append({"name": names[k], "x": round((x + 0.5) / cells.shape[1], 4),
                              "y": round((y + 0.5) / cells.shape[0], 4), "cells": int(region.sum())})
        Image.fromarray(colored).save(cfg.out / f"map-{n}.png")
        (cfg.out / f"names-{n}.json").write_text(json.dumps({"names": spots}, indent=1))
        boxed = scene.copy()
        top, left, bottom, right = (round((v + 1) * side / 2) for v in (row - s, col - s, row + s, col + s))
        width = 8
        ImageDraw.Draw(boxed).rectangle([left + width // 2, top + width // 2, right - 1 - width // 2, bottom - 1 - width // 2],
                                        outline=GLIMPSE_BLUE, width=width)
        boxed.save(cfg.out / f"scene-{n}.png")
        log.info("call %d: named %s", n, ", ".join(s["name"] for s in spots))


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
