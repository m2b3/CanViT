"""Glimpses of the cat photograph used throughout the CanViT project (canvit-pytorch/test_data/Cat03.jpg), taken as
CanViT takes them: the photograph made a 512 px scene (canvit_pytorch.preprocess) and each glimpse sampled from it
at its viewpoint (sample_at_viewpoint, 128 px). "vit" is the glimpse of "Processing a glimpse with a Vision
Transformer", on the cat's head with background around it (chosen by eye among a few crops for how clearly the patch
tokens separate the cat from the background, 2026-10-25), with DINOv3 ViT-B/16's patch tokens on it; "zoom" is the
small glimpse on the cat's eye of the SR-RoPE slide (set by eye). Writes <out>/cat-glimpses.npz."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from canvit_pytorch import Viewpoint, sample_at_viewpoint
from canvit_pytorch.preprocess import imagenet_denormalize, preprocess
from canvit_pytorch.teacher import TEACHER_REPO, load_teacher
from canvit_pytorch.viz.released_model import GLIMPSE_SIZE_PX
from PIL import Image

from experiments.ade20k import SCENE_PX
from experiments.outputs import WORK

SCENE = Path(__file__).parents[5] / "canvit-pytorch/test_data/Cat03.jpg"


@dataclass(frozen=True)
class Config:
    vit: tuple[float, float, float] = (-0.25, -0.0625, 0.75)
    """the ViT slide's glimpse: center (row, col) in [-1, 1] and scale"""
    zoom: tuple[float, float, float] = (-0.12, -0.14, 0.2)
    """the SR-RoPE slide's small glimpse, on the cat's eye"""
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    out: Path = WORK / "vit_glimpse"


def main(cfg: Config) -> None:
    scene = preprocess(SCENE_PX)(Image.open(SCENE).convert("RGB"))[None]  # [1, 3, 512, 512], normalized
    viewpoints = {"vit": cfg.vit, "zoom": cfg.zoom}
    glimpses = {name: sample_at_viewpoint(spatial=scene, viewpoint=Viewpoint(centers=torch.tensor([[row, col]]), scales=torch.tensor([s])),
                                          glimpse_size_px=GLIMPSE_SIZE_PX)
                for name, (row, col, s) in viewpoints.items()}
    teacher = load_teacher(TEACHER_REPO, torch.device(cfg.device))
    with torch.inference_mode():
        tokens = teacher(glimpses["vit"].to(cfg.device)).patches[0].float().cpu().numpy()
    as_uint8 = lambda image: (imagenet_denormalize(image[0]).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    cfg.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cfg.out / "cat-glimpses.npz", scene=as_uint8(scene), tokens=tokens.astype(np.float32), model=TEACHER_REPO,
                        **{f"glimpse_{name}": as_uint8(g) for name, g in glimpses.items()},
                        **{f"viewpoint_{name}": np.array(v) for name, v in viewpoints.items()})


if __name__ == "__main__":
    main(tyro.cli(Config))
