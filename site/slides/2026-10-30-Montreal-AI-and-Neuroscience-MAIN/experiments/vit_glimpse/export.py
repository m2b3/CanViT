"""A plain Vision Transformer on a single glimpse, for "Processing a glimpse with a Vision Transformer": DINOv3 ViT-B/16
on a crop of the cat photograph used throughout the CanViT project (canvit-pytorch/test_data/Cat03.jpg), resized to
the talk's glimpse size. The crop, on the cat's head with background around it, was chosen among a few by eye for how
clearly the patch tokens separate the cat from the background (a sweep of four crops at 128 and 224 px, 2026-10-25).
Writes <out>/cat-glimpse.npz: the scene, the crop box and the patch tokens."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from canvit_pytorch.preprocess import imagenet_normalize
from canvit_pytorch.teacher import TEACHER_REPO, load_teacher
from PIL import Image

from experiments.outputs import WORK

SCENE = Path(__file__).parents[5] / "canvit-pytorch/test_data/Cat03.jpg"


@dataclass(frozen=True)
class Config:
    box: tuple[int, int, int, int] = (150, 0, 1350, 1200)
    """left, top, right, bottom of the glimpse in the photograph's pixels (square)"""
    glimpse_px: int = 128
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    out: Path = WORK / "vit_glimpse"


def main(cfg: Config) -> None:
    left, top, right, bottom = cfg.box
    assert right - left == bottom - top, f"the glimpse must be square, got {cfg.box}"
    scene = Image.open(SCENE).convert("RGB")
    glimpse = scene.crop(cfg.box).resize((cfg.glimpse_px, cfg.glimpse_px), Image.Resampling.BICUBIC)
    pixels = torch.from_numpy(np.array(glimpse)).permute(2, 0, 1).float() / 255
    teacher = load_teacher(TEACHER_REPO, torch.device(cfg.device))
    with torch.inference_mode():
        tokens = teacher(imagenet_normalize(pixels)[None].to(cfg.device)).patches[0].float().cpu().numpy()
    cfg.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cfg.out / "cat-glimpse.npz", scene=np.array(scene), glimpse=np.array(glimpse),
                        box=np.array(cfg.box), tokens=tokens.astype(np.float32), model=TEACHER_REPO)


if __name__ == "__main__":
    main(tyro.cli(Config))
