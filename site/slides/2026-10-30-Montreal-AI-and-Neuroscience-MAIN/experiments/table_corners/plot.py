"""Draw exported examples (export.py) from the saved logits; no model is run here. Panels: scene, truth, dinov3
(DINOv3's labels per glimpse, pasted in place), prob_dinov3 (DINOv3's probability of the object's class per glimpse,
pasted in place, blank elsewhere), a, b, ab (CanViT's labels after A, B, A then B), prob_a, prob_b, prob_ab (CanViT's
probability of the object's class), logit_a, logit_b, logit_ab (the class's logit, one scale for the three),
entropy_a, entropy_b, entropy_ab. Writes a contact sheet per export, or with --separate each panel as
<dir>/<export>/<panel>.png and legend.json: the classes covering at least LEGEND_MIN of CanViT's maps after A, B or A
then B, with their colors."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from canvit_pytorch.benchmarks.ade20k import CLASS_NAMES, IGNORE_LABEL
from canvit_pytorch.specialize.ade20k.figures import LABEL_COLORS, colorize
from PIL import Image, ImageDraw

from experiments import logs
from experiments.glimpses import box_px
from experiments.outputs import WORK

log = logging.getLogger(__name__)

LEGEND_MIN = 0.02
# How logits reach pixel resolution: "nearest" draws what each canvas cell or glimpse patch predicts, its grid
# visible; "bilinear" is the paper's evaluation. Both stay available [Yohaï, 2026-10-01: "keep both possible"].
UPSAMPLING = {"nearest": dict(mode="nearest-exact"), "bilinear": dict(mode="bilinear", align_corners=False)}


@dataclass(frozen=True)
class Config:
    exports: tuple[Path, ...] = ()
    """npz files; every file in the export directory by default"""
    export_dir: Path = WORK / "table_corners/exports"
    panels: tuple[str, ...] = ("scene", "truth", "dinov3", "a", "ab", "prob_a", "prob_ab", "entropy_ab")
    cmap: str = "viridis"
    """matplotlib colormap of p(class) and the class logit"""
    prob_scale: Literal["linear", "log"] = "linear"
    prob_min: float = 0.0
    prob_max: float = 1.0
    log_floor: float = -3.0
    """log10 p shown as the colormap's low end"""
    gamma: float = 1.0
    """applied after scaling to [0, 1]; < 1 lifts weak values"""
    entropy_cmap: str = "magma"
    logits_upsampling: Literal["nearest", "bilinear"] = "nearest"
    box: str = "#2d6cdf"
    """glimpse box color; 'none' for no boxes"""
    blank: str = "#ebebf0"
    """where a map has no prediction"""
    size: int = 256
    """panel side, px"""
    separate: Path | None = None
    """write each panel as <dir>/<export>/<panel>.png instead of one sheet"""
    sheets: Path = field(default=WORK / "table_corners/figures")


def upsampled(logits: np.ndarray, size: int | tuple[int, int], cfg: Config) -> torch.Tensor:
    size = (size, size) if isinstance(size, int) else size
    return F.interpolate(torch.from_numpy(logits.astype(np.float32))[None], size=size,
                         **UPSAMPLING[cfg.logits_upsampling])[0]


def segmentation(labels: np.ndarray, blank: tuple[int, int, int]) -> np.ndarray:
    rgb = colorize(np.where(labels == IGNORE_LABEL, 0, labels)[None])[0]
    rgb[labels == IGNORE_LABEL] = blank
    return rgb


def scalar(values: np.ndarray, cmap: str, vmax: float, vmin: float = 0.0, gamma: float = 1.0) -> np.ndarray:
    """values mapped linearly from [vmin, vmax] to [0, 1], raised to gamma, through a matplotlib colormap."""
    unit = np.clip((values - vmin) / (vmax - vmin), 0, 1) ** gamma
    return (matplotlib.colormaps[cmap](unit)[..., :3] * 255).astype(np.uint8)


def probability(p: np.ndarray, cfg: Config) -> np.ndarray:
    """p(class) on a fixed scale: linear in [prob_min, prob_max], or log10 p in [log_floor, 0]."""
    if cfg.prob_scale == "log":
        return scalar(np.log10(np.clip(p, 1e-12, 1)), cfg.cmap, 0.0, cfg.log_floor, cfg.gamma)
    return scalar(p, cfg.cmap, cfg.prob_max, cfg.prob_min, cfg.gamma)


def glimpses_shown(panel: str) -> tuple[int, ...]:
    """The viewpoints whose boxes a panel draws: those its model saw (the conditions a and b saw one glimpse each)."""
    condition = panel.rsplit("_", 1)[-1]
    return {"a": (0,), "b": (1,)}.get(condition, (0, 1))


def panels(data: dict, meta: dict, cfg: Config) -> dict[str, np.ndarray]:
    S, cls = meta["scene_px"], meta["cls"]
    blank = tuple(int(cfg.blank[i:i + 2], 16) for i in (1, 3, 5))
    out = {"scene": data["scene"], "truth": segmentation(data["labels"], blank)}
    pasted = np.full((S, S), IGNORE_LABEL, dtype=np.int64)
    pasted_p = np.full((S, S), np.nan)
    for key, vp in zip(("dinov3_a", "dinov3_b"), data["viewpoints"]):
        top, left, bottom, right = box_px(vp, S)
        logits = upsampled(data[key], (bottom - top, right - left), cfg)
        inside = (slice(max(top, 0) - top, None), slice(max(left, 0) - left, None))  # the part within the scene
        pasted[max(top, 0):bottom, max(left, 0):right] = logits.argmax(0).numpy()[inside]
        pasted_p[max(top, 0):bottom, max(left, 0):right] = logits.softmax(0)[cls].numpy()[inside]
    out["dinov3"] = segmentation(pasted, blank)
    out["prob_dinov3"] = probability(np.nan_to_num(pasted_p), cfg)
    out["prob_dinov3"][np.isnan(pasted_p)] = blank
    # The class's logit, on one scale for the three conditions, so they compare: the probe's evidence for the class
    # alone, without the softmax's competition between classes.
    class_logits = {cond: upsampled(data[f"canvit_{cond}"], S, cfg)[cls].numpy() for cond in ("a", "b", "ab")}
    low = min(float(v.min()) for v in class_logits.values())
    high = max(float(v.max()) for v in class_logits.values())
    for cond, values in class_logits.items():
        out[f"logit_{cond}"] = scalar(values - low, cfg.cmap, high - low)
    for cond in ("a", "b", "ab"):
        logits = upsampled(data[f"canvit_{cond}"], S, cfg)
        out[cond] = segmentation(logits.argmax(0).numpy(), blank)
        probs = logits.softmax(0)
        out[f"prob_{cond}"] = probability(probs[cls].numpy(), cfg)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(0).numpy()
        out[f"entropy_{cond}"] = scalar(entropy, cfg.entropy_cmap, float(np.log(probs.shape[0])))
    return out


def legend(data: dict, meta: dict, cfg: Config) -> list[dict[str, str]]:
    maps = [upsampled(data[f"canvit_{cond}"], meta["scene_px"], cfg).argmax(0).numpy() for cond in ("a", "b", "ab")]
    share = {int(k): max(float((m == k).mean()) for m in maps) for k in np.unique(np.concatenate([m.ravel() for m in maps]))}
    return [{"class": CLASS_NAMES[k], "color": "#" + bytes(LABEL_COLORS[k].tolist()).hex()}
            for k in sorted(share, key=lambda k: -share[k]) if share[k] >= LEGEND_MIN]


def main(cfg: Config) -> None:
    files = cfg.exports or tuple(sorted(cfg.export_dir.glob("*.npz")))
    assert files, f"no exports in {cfg.export_dir}: run export.py first"
    for path in files:
        data = dict(np.load(path))
        meta = json.loads(str(data.pop("meta")))
        drawn = panels(data, meta, cfg)
        tiles = []
        for name in cfg.panels:
            resample = Image.Resampling.LANCZOS if name == "scene" else Image.Resampling.NEAREST
            tile = Image.fromarray(drawn[name]).resize((cfg.size, cfg.size), resample)
            draw = ImageDraw.Draw(tile)
            for vp in (data["viewpoints"][list(glimpses_shown(name))] if cfg.box != "none" else []):
                top, left, bottom, right = box_px(vp, cfg.size)
                draw.rectangle([left, top, right, bottom], outline=cfg.box, width=max(2, cfg.size // 90))
            tiles.append(tile)
        if cfg.separate:
            target = cfg.separate / path.stem
            target.mkdir(parents=True, exist_ok=True)
            for name, tile in zip(cfg.panels, tiles):
                tile.save(target / f"{name}.png")
            entries = legend(data, meta, cfg)
            (target / "legend.json").write_text(json.dumps(entries, indent=1))
            log.info("%s  legend: %s", target, ", ".join(e["class"] for e in entries))
            continue
        gap = cfg.size // 24
        sheet = Image.new("RGB", (len(tiles) * (cfg.size + gap) - gap, cfg.size), "white")
        for i, tile in enumerate(tiles):
            sheet.paste(tile, (i * (cfg.size + gap), 0))
        cfg.sheets.mkdir(parents=True, exist_ok=True)
        target = cfg.sheets / f"{path.stem}.png"
        sheet.save(target)
        log.info("%s  middle: A %.2f  B %.2f  AB %.2f", target, meta["mid_a"], meta["mid_b"], meta["mid_ab"])


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
