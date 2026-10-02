"""Draw the exports of export.py, without running a model: the teacher's feature map of the scene in PCA colors, and
CanViT's prediction of it after each glimpse in the same basis and color limits, so the colors compare. Panels per
glimpse t: the scene with the glimpse boxes up to t, the glimpse, the prediction, the error (1 − cosine similarity to
the teacher, per patch, on a fixed scale). Writes a contact sheet per export and sequence, or with --separate the panels
under <dir>/<image>/<sequence>/ and cosine.json: the mean cosine similarity after each glimpse over all patches and
over the patches inside none of the glimpses."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import tyro
from PIL import Image, ImageDraw

from experiments import logs
from experiments.distillation.measures import patch_cosines, pca_colors, teacher_pca
from experiments.distillation.sequences import never_seen_patches
from experiments.outputs import WORK

log = logging.getLogger(__name__)

MAPS = {"prediction", "error", "teacher"}  # patch-grid maps: enlarged by nearest neighbor only; photographs smoothly


@dataclass(frozen=True)
class Config:
    exports: tuple[Path, ...] = ()
    """npz files; every file in the export directory by default"""
    export_dir: Path = WORK / "distillation/exports"
    sequences: tuple[str, ...] = ()
    """every sequence in the export when empty"""
    size: int = 256
    """panel side in the contact sheets, px"""
    box: str = "#2d6cdf"
    error_cmap: str = "magma"
    error_max: float = 0.6
    """1 − cosine shown as the colormap's top"""
    separate: Path | None = None
    """write uncomposited panels under DIR/<image>/"""
    sheets: Path = WORK / "distillation/figures"


def boxes(scene: np.ndarray, viewpoints: np.ndarray, upto: int, color: str, width: int) -> Image.Image:
    image = Image.fromarray(scene)
    draw = ImageDraw.Draw(image)
    side = scene.shape[0]
    for i, (row, col, s) in enumerate(viewpoints[:upto + 1]):
        top, left = (row - s + 1) * side / 2, (col - s + 1) * side / 2
        bottom, right = (row + s + 1) * side / 2, (col + s + 1) * side / 2
        draw.rectangle([left, top, right, bottom], outline=color, width=width if i == upto else max(1, width // 2))
    return image


def main(cfg: Config) -> None:
    for path in cfg.exports or tuple(sorted(cfg.export_dir.glob("*.npz"))):
        data = dict(np.load(path))
        meta = json.loads(str(data.pop("meta")))
        grid = meta["grid"]
        teacher = data["teacher_patches"].astype(np.float64)
        basis, limits = teacher_pca(teacher)
        target = pca_colors(teacher, basis, limits).reshape(grid, grid, 3)
        out = cfg.separate / meta["image_id"] if cfg.separate else None
        if out:
            out.mkdir(parents=True, exist_ok=True)
            Image.fromarray(data["scene"]).save(out / "scene.png")
            Image.fromarray(target).save(out / "teacher.png")
        for name in cfg.sequences or meta["sequences"]:
            viewpoints = data[f"{name}/viewpoints"]
            predicted = data[f"{name}/predicted_patches"].astype(np.float64)
            cosine = patch_cosines(predicted, teacher)
            never = never_seen_patches(viewpoints)
            rows = []
            for t, tokens in enumerate(predicted):
                error = (matplotlib.colormaps[cfg.error_cmap](np.clip((1 - cosine[t]) / cfg.error_max, 0, 1))[..., :3]
                         * 255).astype(np.uint8).reshape(grid, grid, 3)
                panels = {"scene": boxes(data["scene"], viewpoints, t, cfg.box, 4),
                          "glimpse": Image.fromarray(data[f"{name}/glimpses"][t]),
                          "prediction": Image.fromarray(pca_colors(tokens, basis, limits).reshape(grid, grid, 3)),
                          "error": Image.fromarray(error)}
                if out:
                    (out / name).mkdir(exist_ok=True)
                    for key, image in panels.items():
                        image.save(out / name / f"{key}-{t}.png")
                rows.append(list(panels.items()) + [("teacher", Image.fromarray(target))])
            similarity = {"all": cosine.mean(1).round(4).tolist(),
                          "never_seen": cosine[:, never].mean(1).round(4).tolist() if never.any() else None,
                          "never_seen_patches": int(never.sum())}
            if out:
                (out / name / "cosine.json").write_text(json.dumps(similarity))
                log.info("%s %s", out / name, similarity)
                continue
            gap = cfg.size // 24
            sheet = Image.new("RGB", (5 * (cfg.size + gap), len(rows) * (cfg.size + gap)), "white")
            for r, row in enumerate(rows):
                for c, (key, image) in enumerate(row):
                    resample = Image.NEAREST if key in MAPS else Image.BILINEAR
                    sheet.paste(image.resize((cfg.size, cfg.size), resample), (c * (cfg.size + gap), r * (cfg.size + gap)))
            cfg.sheets.mkdir(parents=True, exist_ok=True)
            target_path = cfg.sheets / f"{meta['image_id']}-{name}.png"
            sheet.save(target_path)
            log.info("%s %s", target_path, similarity)


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
