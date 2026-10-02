"""Draw exported small objects (export.py) from the saved data; no model is run here. Panels: scene (with the zoomed
glimpse's box), glimpse_full and glimpse_zoom (the 128 px glimpses CanViT received, upscaled nearest-neighbor so their
pixels show), full_in_box (the full-scene glimpse's pixels inside the zoomed glimpse's box: what the zoomed-out view
held of the object), truth, seg_f, seg_fz, seg_ff (CanViT's labels after F, FZ, FF), prob_f, prob_fz, prob_ff
(CanViT's probability of the object's class); NAME_in_box crops a scene-frame map to the zoomed glimpse's box. Writes a
contact sheet per export, or with --separate each panel as <dir>/<export>/<panel>.png and meta.json."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from canvit_pytorch.benchmarks.ade20k import IGNORE_LABEL
from canvit_pytorch.specialize.ade20k.figures import colorize
from PIL import Image, ImageDraw, ImageFont

from experiments import logs
from experiments.glimpses import box_px
from experiments.outputs import WORK

log = logging.getLogger(__name__)

GLIMPSE_PANELS = {"glimpse_full", "glimpse_zoom", "full_in_box"}  # in a glimpse's frame; every other in the scene's
# How logits reach pixel resolution: "nearest" draws each canvas cell's prediction with its grid visible; "bilinear"
# is the paper's evaluation. Both stay available [Yohaï, 2026-10-01: "keep both possible"].
UPSAMPLING = {"nearest": dict(mode="nearest-exact"), "bilinear": dict(mode="bilinear", align_corners=False)}


@dataclass(frozen=True)
class Config:
    exports: tuple[Path, ...] = ()
    """npz files; every file in the export directory by default"""
    export_dir: Path = WORK / "looking_closer/exports"
    panels: tuple[str, ...] = ("scene", "glimpse_full", "glimpse_zoom", "truth", "seg_f", "seg_fz", "prob_f", "prob_fz")
    cmap: str = "inferno"
    """matplotlib colormap of p(class)"""
    prob_min: float = 0.0
    """p(class) shown as the colormap's low end"""
    prob_max: float = 1.0
    """p(class) shown as the colormap's high end"""
    box: str = "#2d6cdf"
    """zoomed glimpse's box color; 'none' for no box"""
    box_panels: tuple[str, ...] = ("scene",)
    """scene-frame panels that show the box"""
    blank: str = "#ebebf0"
    """unlabeled pixels of the ground truth"""
    logits_upsampling: Literal["nearest", "bilinear"] = "nearest"
    size: int = 256
    """panel side, px"""
    separate: Path | None = None
    """write each panel as <dir>/<export>/<panel>.png instead of one sheet"""
    sheets: Path = WORK / "looking_closer/figures"


def rgb(hex_color: str) -> tuple[int, int, int]:
    return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))


def upsampled(logits: np.ndarray, size: int, cfg: Config) -> torch.Tensor:
    return F.interpolate(torch.from_numpy(logits.astype(np.float32))[None], size=(size, size),
                         **UPSAMPLING[cfg.logits_upsampling])[0]


def segmentation(labels: np.ndarray, blank: tuple[int, int, int]) -> np.ndarray:
    colored = colorize(np.where(labels == IGNORE_LABEL, 0, labels)[None])[0]
    colored[labels == IGNORE_LABEL] = blank
    return colored


def panels(data: dict, meta: dict, cfg: Config) -> dict[str, tuple[np.ndarray, Image.Resampling]]:
    """Each panel at its native resolution, with the resampling that resizes it to the panel side."""
    S, cls = meta["scene_px"], meta["cls"]
    out = {
        "scene": (data["scene"], Image.Resampling.LANCZOS),
        "glimpse_full": (data["glimpse_full"], Image.Resampling.NEAREST),
        "glimpse_zoom": (data["glimpse_zoom"], Image.Resampling.NEAREST),
        "truth": (segmentation(data["labels"], rgb(cfg.blank)), Image.Resampling.NEAREST),
    }
    zoom = next(vp for vp in data["viewpoints"] if vp[2] < 1)
    side = data["glimpse_full"].shape[0]  # the full-scene glimpse spans the whole scene
    top, left, bottom, right = (min(max(v, 0), side) for v in box_px(zoom, side))
    out["full_in_box"] = (np.ascontiguousarray(data["glimpse_full"][top:bottom, left:right]), Image.Resampling.NEAREST)
    for cond in ("f", "fz", "ff"):
        logits = upsampled(data[f"logits_{cond}"], S, cfg)
        out[f"seg_{cond}"] = (segmentation(logits.argmax(0).numpy(), rgb(cfg.blank)), Image.Resampling.NEAREST)
        unit = np.clip((logits.softmax(0)[cls].numpy() - cfg.prob_min) / (cfg.prob_max - cfg.prob_min), 0, 1)
        out[f"prob_{cond}"] = ((matplotlib.colormaps[cfg.cmap](unit)[..., :3] * 255).astype(np.uint8),
                               Image.Resampling.NEAREST)
    # Scene-frame maps cropped to the zoomed glimpse's box, so the object fills the panel as in glimpse_zoom.
    top, left, bottom, right = (min(max(v, 0), S) for v in box_px(zoom, S))
    for name in ("truth", "seg_f", "seg_fz", "seg_ff", "prob_f", "prob_fz", "prob_ff"):
        pixels, resampling = out[name]
        out[f"{name}_in_box"] = (np.ascontiguousarray(pixels[top:bottom, left:right]), resampling)
    return out


def tile(name: str, drawn: dict, data: dict, cfg: Config) -> Image.Image:
    pixels, resampling = drawn[name]
    image = Image.fromarray(pixels).resize((cfg.size, cfg.size), resampling)
    if name in cfg.box_panels and cfg.box != "none":
        draw = ImageDraw.Draw(image)
        for vp in data["viewpoints"]:
            if vp[2] < 1:  # the full-scene glimpse's box is the panel's edge
                top, left, bottom, right = box_px(vp, cfg.size)
                draw.rectangle([left, top, right - 1, bottom - 1], outline=cfg.box, width=max(2, cfg.size // 90))
    return image


def caption(meta: dict) -> str:
    return (f"{meta['image_id']}  {meta['name']}  area {100 * meta['area']:.2f}%  zoom scale {meta['zoom'][2]:.3f}   "
            f"recall F {meta['recall_f']:.2f} -> FZ {meta['recall_fz']:.2f} (FF {meta['recall_ff']:.2f})   "
            f"p(class) F {meta['prob_f']:.2f} -> FZ {meta['prob_fz']:.2f} (FF {meta['prob_ff']:.2f})   "
            f"fp FZ {meta['fp_fz']:.2f}")


def main(cfg: Config) -> None:
    framed = [n for n in cfg.box_panels if n in GLIMPSE_PANELS or n.endswith("_in_box")]
    assert not framed, f"not in the scene's frame, so no box: {framed}"
    files = cfg.exports or tuple(sorted(cfg.export_dir.glob("*.npz")))
    assert files, f"no exports in {cfg.export_dir}: run export.py first"
    for path in files:
        data = dict(np.load(path))
        meta = json.loads(str(data.pop("meta")))
        drawn = panels(data, meta, cfg)
        tiles = [tile(name, drawn, data, cfg) for name in cfg.panels]
        if cfg.separate:
            target = cfg.separate / path.stem
            target.mkdir(parents=True, exist_ok=True)
            for name, image in zip(cfg.panels, tiles):
                image.save(target / f"{name}.png")
            # What a slide needs to place things on these panels: the zoomed viewpoint (row, col, scale in scene
            # coordinates), the object's box in scene fractions (top, left, bottom, right), and the numbers it shows.
            S = meta["scene_px"]
            (target / "meta.json").write_text(json.dumps({
                "image_id": meta["image_id"], "class": meta["name"], "zoom": [float(v) for v in meta["zoom"]],
                "object_box": [v / S for v in meta["bbox"]], "prob_f": meta["prob_f"], "prob_fz": meta["prob_fz"],
                "prob_ff": meta["prob_ff"], "logits_upsampling": cfg.logits_upsampling}, indent=1))
            log.info("%s", target)
            continue
        gap, header = cfg.size // 24, 40
        font = ImageFont.load_default(size=14)
        sheet = Image.new("RGB", (len(tiles) * (cfg.size + gap) - gap, cfg.size + header), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((0, 2), caption(meta), fill="black", font=font)
        for i, (name, image) in enumerate(zip(cfg.panels, tiles)):
            draw.text((i * (cfg.size + gap), header - 18), name, fill="black", font=font)
            sheet.paste(image, (i * (cfg.size + gap), header))
        cfg.sheets.mkdir(parents=True, exist_ok=True)
        target = cfg.sheets / f"{path.stem}.png"
        sheet.save(target)
        log.info("%s  %s", target, caption(meta))


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
