"""The "DINOv3 feature maps" slide's images, from export.py's and decode.py's exports, into <out>/<id>/, every map
upsampled with nearest neighbor so its patches stay visible:
  photo.png           the scene as DINOv3 saw it
  pca.png             patch features, LayerNormed, projected on the principal components COMPONENTS, each clipped to
                      its CLIP percentiles: chosen among variants (variants.py) for how distinct the objects look
  prob-<class>.png    the ADE20K linear probe's probability of the class per patch, inferno from 0 to 1
  maps.json           what each image is"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from canvit_pytorch.benchmarks.ade20k import CLASS_NAMES
from canvit_pytorch.hub.repos import released_dinov3_ade20k_probe
from matplotlib import colormaps
from PIL import Image

from experiments import logs
from experiments.foundation.decode import PROBE_INPUT_PX
from experiments.foundation.pca_colors import colors, components
from experiments.outputs import DECK_DATA, WORK

log = logging.getLogger(__name__)

COMPONENTS = (1, 2, 3)  # the second to fourth principal components
CLIP = (2.0, 98.0)


@dataclass(frozen=True)
class Config:
    image_id: str
    short_side: int
    classes: tuple[str, ...]
    exports: Path = WORK / "foundation/exports"
    out: Path = DECK_DATA / "foundation"


def nearest(grid: np.ndarray, size: tuple[int, int]) -> Image.Image:
    return Image.fromarray(grid).resize(size, Image.Resampling.NEAREST)


def main(cfg: Config) -> None:
    data = np.load(cfg.exports / f"{cfg.image_id}-{cfg.short_side}.npz")
    probs = np.load(cfg.exports / f"{cfg.image_id}-{cfg.short_side}-probs.npz")["probs"].astype(np.float32)
    grid_h, grid_w = (int(v) for v in data["grid"])
    photo = Image.fromarray(data["photo"])
    out = cfg.out / cfg.image_id
    out.mkdir(parents=True, exist_ok=True)
    photo.save(out / "photo.png")
    rgb = colors(components(data["features"].astype(np.float64), max(COMPONENTS) + 1)[:, list(COMPONENTS)], *CLIP)
    nearest((rgb.reshape(grid_h, grid_w, 3) * 255).astype(np.uint8), photo.size).save(out / "pca.png")
    inferno = colormaps["inferno"]
    for name in cfg.classes:
        p = probs[CLASS_NAMES.index(name)]
        nearest((inferno(p)[..., :3] * 255).astype(np.uint8), photo.size).save(out / f"prob-{name.replace(' ', '-')}.png")
    (out / "maps.json").write_text(json.dumps({
        "image_id": cfg.image_id, "input_px": list(photo.size), "patch_grid": [grid_h, grid_w],
        "pca": {"components": [c + 1 for c in COMPONENTS], "clip_percentiles": list(CLIP)},
        "probe": released_dinov3_ade20k_probe("dv3b", input_size_px=PROBE_INPUT_PX),
        "classes": list(cfg.classes), "colormap": "inferno, p from 0 to 1"}, indent=1))
    log.info("wrote %s: %s", out, ", ".join(sorted(p.name for p in out.iterdir())))


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
