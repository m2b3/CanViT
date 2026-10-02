"""Save, uncolored, everything a figure of chosen examples needs: the scene, its labels, the two viewpoints, CanViT's
logits after A, after B and after A then B, and DINOv3's logits on each glimpse alone (float16). plot.py colors them."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from canvit_pytorch.benchmarks.ade20k import CLASS_NAMES
from canvit_pytorch.hub import repos
from canvit_pytorch.probes import SegmentationProbe
from canvit_pytorch.teacher import DINOV3_REPOS, load_teacher
from canvit_pytorch.viz.released_model import CANVAS_GRID_SIZE, load_released_segmenter

from experiments import logs
from experiments.ade20k import SCENE_PX, load, pixels
from experiments.glimpses import GLIMPSE_PX
from experiments.outputs import WORK
from experiments.table_corners.definition import Example, canvit_logits, dinov3_glimpse_logits

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    ids: tuple[str, ...] = ()
    """IMAGE_ID:CLASS pairs from the sweep's examples"""
    top: int = 0
    """or the sweep's best examples: the unseen middle labeled after both ends beyond what one end gives, few false
    positives between the glimpses"""
    examples: Path = WORK / "table_corners/examples.json"
    out: Path = WORK / "table_corners/exports"
    device: str = "mps"


def chosen(cfg: Config, examples: list[Example]) -> list[Example]:
    if cfg.ids:
        wanted = [tuple(i.split(":")) for i in cfg.ids]
        picked = [ex for image_id, name in wanted for ex in examples if ex.image_id == image_id and ex.name == name]
        assert len(picked) == len(wanted), f"not all of {cfg.ids} are in {cfg.examples}"
        return picked
    ranked = sorted((ex for ex in examples if ex.fp_ab < 0.3), key=lambda ex: ex.mid_ab - 0.5 * max(ex.mid_a, ex.mid_b),
                    reverse=True)
    return ranked[:cfg.top]


@torch.inference_mode()
def main(cfg: Config) -> None:
    examples = [Example(**{**e, "a": tuple(e["a"]), "b": tuple(e["b"])}) for e in json.loads(cfg.examples.read_text())]
    picked = chosen(cfg, examples)
    assert picked, "nothing to export: pass --ids or --top"
    device = torch.device(cfg.device)
    model = load_released_segmenter(scene_size_px=SCENE_PX, canvas_grid_size=CANVAS_GRID_SIZE, device=device).model
    teacher = load_teacher(DINOV3_REPOS["vitb16"], device)
    probe_repo = repos.released_dinov3_ade20k_probe("dv3b", input_size_px=GLIMPSE_PX)
    probe = SegmentationProbe.from_pretrained(probe_repo).to(device).eval()
    cfg.out.mkdir(parents=True, exist_ok=True)
    f16 = lambda t: t.cpu().numpy().astype(np.float16)  # noqa: E731
    for ex in picked:
        image, labels = load(ex.image_id)
        images = image[None].to(device)
        path = cfg.out / f"{ex.image_id}-{ex.name.replace(' ', '_')}.npz"
        np.savez_compressed(
            path,
            scene=pixels(image),
            labels=labels.astype(np.uint8),
            viewpoints=np.array([ex.a, ex.b], dtype=np.float32),
            canvit_a=f16(canvit_logits(model, images, [[ex.a]])[0]),
            canvit_b=f16(canvit_logits(model, images, [[ex.b]])[0]),
            canvit_ab=f16(canvit_logits(model, images, [[ex.a], [ex.b]])[0]),
            dinov3_a=f16(dinov3_glimpse_logits(teacher, probe, image, ex.a)),
            dinov3_b=f16(dinov3_glimpse_logits(teacher, probe, image, ex.b)),
            meta=np.array(json.dumps({**ex.__dict__, "class_names": list(CLASS_NAMES), "scene_px": SCENE_PX,
                                      "glimpse_px": GLIMPSE_PX})),
        )
        log.info("%s", path)


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
