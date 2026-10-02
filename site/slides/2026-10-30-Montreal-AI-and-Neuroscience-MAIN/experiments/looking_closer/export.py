"""Save, uncolored, everything a figure of chosen small objects needs: the scene, its labels, the object's mask, the
two viewpoints, both glimpses as CanViT received them, and CanViT's logits after F, FZ and FF (float16). plot.py
colors them."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from canvit_pytorch.benchmarks.ade20k import CLASS_NAMES
from canvit_pytorch.viz.released_model import CANVAS_GRID_SIZE, load_released_segmenter

from experiments import logs
from experiments.ade20k import SCENE_PX, load, pixels
from experiments.glimpses import FULL_SCENE, GLIMPSE_PX, glimpse_pixels
from experiments.looking_closer.definition import canvit_logits, scene_maps, small_objects
from experiments.outputs import WORK

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    ids: tuple[str, ...] = ()
    """IMAGE_ID:CLASS pairs from the sweep's objects"""
    top: int = 0
    """or the sweep's objects labeled after the zoom but not after the full-scene glimpse: small, whole (not cut by
    the scene's border), the only sizeable instance of its class, with few false positives after the zoom"""
    max_fp: float = 0.5
    """of fp_fz, false-positive pixels over the object's area"""
    max_area: float = 0.01
    """of the scene"""
    min_share: float = 0.5
    """the object's share of its class's pixels"""
    examples: Path = WORK / "looking_closer/examples.json"
    out: Path = WORK / "looking_closer/exports"
    device: str = "mps"


def chosen(cfg: Config, rows: list[dict]) -> list[dict]:
    if cfg.ids:
        wanted = [tuple(i.split(":")) for i in cfg.ids]
        picked = [r for image_id, name in wanted for r in rows if r["image_id"] == image_id and r["name"] == name]
        assert len(picked) == len(wanted), f"not all of {cfg.ids} are in {cfg.examples}"
        return picked
    eligible = [r for r in rows if r["fp_fz"] <= cfg.max_fp and r["area"] <= cfg.max_area and not r["touches_border"]
                and r["area"] / r["class_area"] >= cfg.min_share]
    return sorted(eligible, key=lambda r: r["recall_fz"] - r["recall_f"], reverse=True)[:cfg.top]


@torch.inference_mode()
def main(cfg: Config) -> None:
    picked = chosen(cfg, json.loads(cfg.examples.read_text()))
    assert picked, "nothing to export: pass --ids or --top"
    device = torch.device(cfg.device)
    model = load_released_segmenter(scene_size_px=SCENE_PX, canvas_grid_size=CANVAS_GRID_SIZE, device=device).model
    cfg.out.mkdir(parents=True, exist_ok=True)
    f16 = lambda t: t.cpu().numpy().astype(np.float16)  # noqa: E731
    for row in picked:
        image, labels = load(row["image_id"])
        (obj, mask), = [(o, m) for o, m in small_objects(row["image_id"], labels) if o.cls == row["cls"]]
        assert list(obj.zoom) == row["zoom"], f"{row['image_id']}:{row['name']}: zoom {obj.zoom} != the sweep's {row['zoom']}"
        logits = {cond: t[0] for cond, t in canvit_logits(model, image[None].to(device), [obj.zoom]).items()}
        for cond in ("f", "fz"):
            pred, _ = scene_maps(logits[cond][None], [obj.cls])
            recall = float((pred[0][mask] == obj.cls).mean())
            assert abs(recall - row[f"recall_{cond}"]) < 0.05, (
                f"{row['image_id']}:{row['name']}: recall {cond} {recall:.3f} here, {row[f'recall_{cond}']:.3f} in the sweep")
        path = cfg.out / f"{obj.image_id}-{obj.name.replace(' ', '_')}.npz"
        np.savez_compressed(
            path,
            scene=pixels(image),
            labels=labels.astype(np.uint8),
            object=mask,
            viewpoints=np.array([FULL_SCENE, obj.zoom], dtype=np.float32),
            glimpse_full=glimpse_pixels(image, FULL_SCENE, device),
            glimpse_zoom=glimpse_pixels(image, obj.zoom, device),
            logits_f=f16(logits["f"]),
            logits_fz=f16(logits["fz"]),
            logits_ff=f16(logits["ff"]),
            meta=np.array(json.dumps({**row, "class_names": list(CLASS_NAMES), "scene_px": SCENE_PX,
                                      "glimpse_px": GLIMPSE_PX})),
        )
        log.info("%s  %s  area %.2f%%  recall F %.2f FZ %.2f FF %.2f  fp FZ %.2f", path, row["name"], 100 * row["area"],
                 row["recall_f"], row["recall_fz"], row["recall_ff"], row["fp_fz"])


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
