"""Save, uncolored, what a figure of chosen three-object sequences of the sweep needs, recomputed with the model. Per
sequence, into <out>/<image id>-<class>-<class>-<class>/:
  scene.png                    the scene as CanViT sees it (the short side resized, center-cropped)
  scene_t<t>.png               the scene with the boxes of glimpses 0..t (glimpse t BOX_WIDTH px, earlier ones thinner)
  <class>_kept_t<t>.png        p(class) per canvas cell after glimpse t, the canvas kept: softmax of the probe's
                               G x G logits, G x G 16-bit gray, p = value / 65535; upsample nearest only to display
  <class>_reset_t<t>.png       the same, the canvas reset before every glimpse (t = 0 equals kept)
  <class>_mask.png             the object's pixels (8-bit, 255 on the object)
  memory.json                  viewpoints, objects, per-glimpse means of p over each object, and what the files hold
Means over an object, both kept in memory.json: mean_p_<condition>_cells from the exported maps (each cell's p over
its pixels), mean_p_<condition>_bilinear as the sweep scored them (logits upsampled bilinearly, then softmax)."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from canvit_pytorch.hub.repos import FLAGSHIP, released_ade20k_probe
from canvit_pytorch.viz.released_model import CANVAS_GRID_SIZE, load_released_segmenter
from PIL import Image, ImageDraw

from experiments import logs
from experiments.ade20k import SCENE_PX, load, pixels
from experiments.glimpses import GLIMPSE_PX
from experiments.memory.definition import (
    CONDITIONS,
    Geometry,
    candidates,
    cell_probabilities,
    class_probabilities,
    covering_box_px,
    nearest_to_scene,
    rollouts,
    slug,
)
from experiments.outputs import DECK_DATA, WORK

log = logging.getLogger(__name__)

TOLERANCE = 0.01  # p over an object here against the sweep's: batch composition shifts MPS numerics slightly
VIEWPOINT_TOLERANCE = 1e-6  # the sweep rounds viewpoints to 6 decimals
GLIMPSE_BLUE = "#2d6cdf"  # --glimpse in site/css/canvit.css
BOX_WIDTH, EARLIER_BOX_WIDTH = 4, 2  # px at the scene's size
P_SCALE = 65535  # 16-bit PNG value of p = 1


@dataclass(frozen=True)
class Config:
    ids: tuple[str, ...]
    """IMAGE_ID:CLASS#COMPONENT,CLASS#COMPONENT,CLASS#COMPONENT in glimpse order, as rank.py prints them"""
    out: Path = DECK_DATA / "memory"
    sweep: Path = WORK / "memory/sweep.json"
    device: str = "mps"


def find(rows: list[dict], wanted: str) -> dict:
    image_id, specs = wanted.split(":")
    objs = [spec.split("#") for spec in specs.split(",")]
    matches = [r for r in rows if r["image_id"] == image_id
               and [(o["name"], str(o["component"])) for o in r["objects"]] == [(n, c) for n, c in objs]]
    assert len(matches) == 1, f"{wanted}: {len(matches)} matching sequences in the sweep"
    return matches[0]


def boxed(scene: np.ndarray, vps: tuple[tuple[float, float, float], ...], t: int) -> Image.Image:
    image = Image.fromarray(scene)
    draw = ImageDraw.Draw(image)
    for i, vp in enumerate(vps[:t + 1]):
        top, left, bottom, right = covering_box_px(vp)
        draw.rectangle([left, top, right - 1, bottom - 1], outline=GLIMPSE_BLUE,
                       width=BOX_WIDTH if i == t else EARLIER_BOX_WIDTH)
    return image


def as_16bit(p: np.ndarray) -> Image.Image:
    assert p.shape == (CANVAS_GRID_SIZE, CANVAS_GRID_SIZE) and p.min() >= 0 and p.max() <= 1, (p.shape, p.min(), p.max())
    return Image.fromarray((p * P_SCALE).round().astype(np.uint16))


def export(model, row: dict, geometry: Geometry, out: Path, device: torch.device) -> Path:
    image_id = row["image_id"]
    image, labels = load(image_id)
    found, _ = candidates(image_id, labels, geometry)
    key = lambda objs: [(o.cls, o.component) for o in objs]  # noqa: E731
    (c,) = [c for c in found if key(c.objs) == [(o["cls"], o["component"]) for o in row["objects"]]]
    assert np.allclose(c.viewpoints, row["viewpoints"], rtol=0, atol=VIEWPOINT_TOLERANCE), \
        f"{image_id}: viewpoints {c.viewpoints} here, {row['viewpoints']} in the sweep"
    logits = rollouts(model, image[None].to(device), [c.viewpoints])
    classes = [o.cls for o in c.objs]
    cells = {cond: [cell_probabilities(step, classes)[0].cpu().numpy() for step in steps]  # [t] -> [3, G, G]
             for cond, steps in logits.items()}
    bilinear = {cond: [class_probabilities(step, classes)[0].cpu().numpy() for step in steps]  # [t] -> [3, S, S]
                for cond, steps in logits.items()}
    names = [o.name.replace(" ", "_") for o in c.objs]
    target = out / slug(image_id, [o.name for o in c.objs])
    target.mkdir(parents=True, exist_ok=True)
    scene = pixels(image)
    files: dict[str, Image.Image] = {"scene.png": Image.fromarray(scene)}
    objects = []
    for i, (o, mask, name) in enumerate(zip(c.objs, c.masks, names)):
        files[f"{name}_mask.png"] = Image.fromarray(mask.astype(np.uint8) * 255)
        entry = {"class": o.name, "file_prefix": name, "class_index": o.cls, "glimpse": i,
                 "component": o.component, "area_of_scene": round(o.area, 4), "bbox_px": list(o.bbox)}
        for cond in CONDITIONS:
            for t, p in enumerate(cells[cond]):
                files[f"{name}_{cond}_t{t}.png"] = as_16bit(p[i])
            entry[f"mean_p_{cond}_cells"] = [round(float(nearest_to_scene(p[i])[mask].mean()), 4) for p in cells[cond]]
            entry[f"mean_p_{cond}_bilinear"] = [round(float(p[i][mask].mean()), 4) for p in bilinear[cond]]
            swept = [step[i] for step in row["prob"][cond]]
            assert np.allclose(entry[f"mean_p_{cond}_bilinear"], swept, atol=TOLERANCE), \
                f"{image_id} {o.name} {cond}: p {entry[f'mean_p_{cond}_bilinear']} here, {swept} in the sweep"
        objects.append(entry)
    for t in range(len(c.viewpoints)):
        files[f"scene_t{t}.png"] = boxed(scene, c.viewpoints, t)
    record = {
        "what": "CanViT-B (released) and its ADE20K probe on a 64x64 canvas: three glimpses, each on an object of a "
                "distinct class, in left-to-right order. Probability maps: p(class) per canvas cell, softmax of the "
                "probe's 64x64 logits over the 150 classes, 64x64 16-bit gray PNG, p = value / 65535; display them "
                "upsampled nearest only. mean_p_<kept|reset>_cells[t]: p averaged over the object's pixels after "
                "glimpse t, each pixel taking its cell's p (what the maps show); mean_p_<kept|reset>_bilinear[t]: the "
                "same from logits upsampled bilinearly to the scene before the softmax (the paper's evaluation, the "
                "sweep's scoring). kept: the canvas carried from glimpse to glimpse; reset: reset before each glimpse. "
                "Viewpoints: (row, col) center in [-1, 1], scale = the glimpse's side over the scene's.",
        "image_id": image_id, "category": row["category"], "short_side_px": row["short_side_px"],
        "scene_px": SCENE_PX, "glimpse_px": GLIMPSE_PX, "canvas_grid": CANVAS_GRID_SIZE,
        "model_repo": FLAGSHIP,
        "probe_repo": released_ade20k_probe("in21k", scene_size_px=SCENE_PX, canvas_grid_size=CANVAS_GRID_SIZE),
        "geometry": asdict(geometry),
        "viewpoints": [{"row": round(r, 4), "col": round(col, 4), "scale": round(s, 4),
                        "box_px": list(covering_box_px((r, col, s)))} for r, col, s in c.viewpoints],
        "objects": objects,
    }
    for name, im in files.items():
        im.save(target / name, optimize=True)
    (target / "memory.json").write_text(json.dumps(record, indent=1))
    return target


@torch.inference_mode()
def main(cfg: Config) -> None:
    sweep = json.loads(cfg.sweep.read_text())
    geometry = Geometry(**sweep["geometry"])
    device = torch.device(cfg.device)
    model = load_released_segmenter(scene_size_px=SCENE_PX, canvas_grid_size=CANVAS_GRID_SIZE, device=device).model
    for wanted in cfg.ids:
        target = export(model, find(sweep["sequences"], wanted), geometry, cfg.out, device)
        log.info("%s  %.1f MB", target, sum(f.stat().st_size for f in target.iterdir() if f.is_file()) / 2**20)


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
