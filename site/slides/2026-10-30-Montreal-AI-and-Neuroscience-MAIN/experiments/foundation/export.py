"""DINOv3 ViT-B/16's patch features of a scene at its own aspect ratio, for the "DINOv3 feature maps" slide, and the
patches whose similarity maps best outline their own class.

The scene is resized (aspect kept, short side --short-side px) to whole 16 px patches. For every patch inside a class
of the annotation, the cosine similarity of every patch to it is thresholded at each of THRESHOLDS; a patch's score is
the best IoU between that set and its class's patches. Writes <out>/<id>-<short side>.npz (patch features, grid,
labels per patch, per-class best patches and scores) and prints the ranking; plot.py draws the slide's images."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from canvit_pytorch.benchmarks.ade20k import CLASS_NAMES, dataset_root, decode_annotation
from canvit_pytorch.preprocess import imagenet_normalize
from canvit_pytorch.teacher import DINOV3_PATCH_SIZE, TEACHER_REPO, load_teacher
from PIL import Image

from experiments import logs
from experiments.outputs import WORK

log = logging.getLogger(__name__)

THRESHOLDS = np.arange(0.2, 0.85, 0.05)
MIN_CLASS_PATCHES = 12


@dataclass(frozen=True)
class Config:
    image_id: str
    short_side: int = 512
    """the scene's short side in pixels (aspect kept, rounded to whole patches)"""
    out: Path = WORK / "foundation/exports"
    device: str = "mps"


def main(cfg: Config) -> None:
    root = dataset_root()
    photo = Image.open(root / "images/validation" / f"{cfg.image_id}.jpg").convert("RGB")
    annotation = Image.open(root / "annotations/validation" / f"{cfg.image_id}.png")
    scale = cfg.short_side / min(photo.size)
    width, height = (round(side * scale / DINOV3_PATCH_SIZE) * DINOV3_PATCH_SIZE for side in photo.size)
    photo = photo.resize((width, height), Image.Resampling.BICUBIC)
    grid_w, grid_h = width // DINOV3_PATCH_SIZE, height // DINOV3_PATCH_SIZE
    # The class of each patch: the annotation's majority label inside it (255 where mostly unlabeled).
    labels = decode_annotation(torch.from_numpy(np.array(annotation.resize((width, height), Image.Resampling.NEAREST)))).numpy()
    blocks = labels.reshape(grid_h, DINOV3_PATCH_SIZE, grid_w, DINOV3_PATCH_SIZE).transpose(0, 2, 1, 3).reshape(grid_h, grid_w, -1)
    patch_labels = np.array([[np.bincount(b, minlength=256).argmax() for b in row] for row in blocks], dtype=np.uint8)

    teacher = load_teacher(TEACHER_REPO, torch.device(cfg.device))
    pixels = torch.from_numpy(np.asarray(photo, dtype=np.float32) / 255.0).permute(2, 0, 1)
    with torch.inference_mode():
        features = teacher(imagenet_normalize(pixels)[None].to(cfg.device)).patches[0].float().cpu().numpy()
    assert features.shape[0] == grid_h * grid_w, (features.shape, grid_h, grid_w)

    unit = features / np.linalg.norm(features, axis=1, keepdims=True)
    similarity = unit @ unit.T  # [N, N]
    flat = patch_labels.reshape(-1)
    best: dict[str, tuple[int, float, float]] = {}
    for cls in np.unique(flat):
        if cls == 255:
            continue
        mask = flat == cls
        if mask.sum() < MIN_CLASS_PATCHES:
            continue
        candidates = np.flatnonzero(mask)
        sets = similarity[candidates][:, :, None] >= THRESHOLDS[None, None, :]  # [K, N, T]
        iou = (sets & mask[None, :, None]).sum(axis=1) / (sets | mask[None, :, None]).sum(axis=1)
        k, t = np.unravel_index(iou.argmax(), iou.shape)
        best[CLASS_NAMES[cls]] = (int(candidates[k]), float(iou[k, t]), float(THRESHOLDS[t]))

    cfg.out.mkdir(parents=True, exist_ok=True)
    path = cfg.out / f"{cfg.image_id}-{cfg.short_side}.npz"
    np.savez_compressed(path, features=features.astype(np.float16), grid=np.array([grid_h, grid_w]),
                        patch_labels=patch_labels, photo=np.asarray(photo),
                        best_names=np.array(list(best)), best=np.array([v for v in best.values()]))
    log.info("%s: %dx%d px, %dx%d patches -> %s", cfg.image_id, width, height, grid_w, grid_h, path)
    for name, (index, iou, threshold) in sorted(best.items(), key=lambda kv: -kv[1][1]):
        row, col = divmod(index, grid_w)
        print(f"  {name:14s} patch (row {row:2d}, col {col:2d})  IoU {iou:.3f} at cos >= {threshold:.2f}  "
              f"class patches {int((flat == CLASS_NAMES.index(name)).sum())}")


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
