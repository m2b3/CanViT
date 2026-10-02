"""Variants of the DINOv3 PCA map for the "DINOv3 feature maps" slide: input resolution x principal components x
contrast, each scored by how much of the colors' variance the annotation's classes explain (eta squared over the
three channels, labeled patches only; higher means objects stand apart in color) and by mean saturation. Writes a
labeled sheet (nearest-neighbor upsampling) and prints the scores."""

import colorsys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from PIL import Image, ImageDraw, ImageFont

from experiments.foundation.pca_colors import colors, components
from experiments.outputs import WORK

TILE = (384, 256)
COMPONENTS = [(0, 1, 2), (1, 2, 3), (0, 2, 3), (0, 1, 3)]
CONTRASTS = {"min-max": (0.0, 100.0), "clip 2-98%": (2.0, 98.0)}


@dataclass(frozen=True)
class Config:
    image_id: str
    short_sides: tuple[int, ...] = (512, 768, 1024)
    exports: Path = WORK / "foundation/exports"
    out: Path = WORK / "foundation/variants.png"


def eta_squared(rgb: np.ndarray, labels: np.ndarray) -> float:
    keep = labels != 255
    rgb, labels = rgb[keep], labels[keep]
    total = ((rgb - rgb.mean(axis=0)) ** 2).sum()
    between = sum(((labels == c).sum() * ((rgb[labels == c].mean(axis=0) - rgb.mean(axis=0)) ** 2).sum()) for c in np.unique(labels))
    return float(between / total)


def main(cfg: Config) -> None:
    font = ImageFont.load_default(size=15)
    tiles = []
    for side in cfg.short_sides:
        data = np.load(cfg.exports / f"{cfg.image_id}-{side}.npz")
        grid_h, grid_w = (int(v) for v in data["grid"])
        labels = data["patch_labels"].reshape(-1)
        projection = components(data["features"].astype(np.float64), 4)
        for comps in COMPONENTS:
            for name, (lo, hi) in CONTRASTS.items():
                rgb = colors(projection[:, comps], lo, hi)
                score = eta_squared(rgb, labels)
                saturation = float(np.mean([colorsys.rgb_to_hsv(*c)[1] for c in rgb]))
                image = Image.fromarray((rgb.reshape(grid_h, grid_w, 3) * 255).astype(np.uint8)).resize(TILE, Image.Resampling.NEAREST)
                caption = f"{side}px PCs {tuple(c + 1 for c in comps)} {name}  eta2 {score:.2f} sat {saturation:.2f}"
                tiles.append((score, caption, image))
                print(caption)
    columns = len(COMPONENTS) * len(CONTRASTS)
    sheet = Image.new("RGB", (columns * (TILE[0] + 6), len(cfg.short_sides) * (TILE[1] + 26)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (_, caption, image) in enumerate(tiles):
        x, y = (i % columns) * (TILE[0] + 6), (i // columns) * (TILE[1] + 26)
        sheet.paste(image, (x, y + 22))
        draw.text((x, y + 2), caption, fill="black", font=font)
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(cfg.out)
    print("best by eta2:", *(c for _, c, _ in sorted(tiles, key=lambda t: -t[0])[:5]), sep="\n  ")


if __name__ == "__main__":
    main(tyro.cli(Config))
