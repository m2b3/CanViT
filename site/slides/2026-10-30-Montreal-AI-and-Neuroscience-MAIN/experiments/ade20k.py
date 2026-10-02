"""ADE20K validation scenes as the experiments give them to CanViT: the photograph's short side resized to the released
scene size and center-cropped ([3, S, S], ImageNet-normalized), its labels the same way ([S, S], 255 unlabeled)."""

import numpy as np
import torch
from canvit_pytorch.benchmarks.ade20k import CLASS_NAMES, NUM_VALIDATION_IMAGES, dataset_root, decode_annotation
from canvit_pytorch.hub.repos import RELEASED_SCENE_SIZE_PX
from canvit_pytorch.preprocess import imagenet_denormalize, preprocess, preprocess_labels
from PIL import Image

SCENE_PX = RELEASED_SCENE_SIZE_PX

# Amorphous regions, surfaces and structures: a piece of one is not an object one looks at.
STUFF = {
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "grass", "sidewalk", "earth", "mountain", "water",
    "sea", "field", "sand", "skyscraper", "path", "runway", "river", "hill", "land", "dirt track", "lake",
    "swimming pool", "waterfall", "rock", "rug", "curtain", "blind", "fence", "railing", "bannister", "stairs",
    "stairway", "step", "house", "hovel", "tower", "bridge", "pier", "grandstand", "stage", "awning", "canopy", "base",
    "column", "windowpane", "countertop", "counter", "bar", "escalator", "conveyer belt", "booth",
}
STUFF_IDS = {CLASS_NAMES.index(name) for name in STUFF}
assert len(STUFF_IDS) == len(STUFF)


def validation_ids() -> list[str]:
    ids = sorted(p.stem for p in (dataset_root() / "images/validation").glob("*.jpg"))
    assert len(ids) == NUM_VALIDATION_IMAGES, f"{dataset_root()}: {len(ids)} validation images, not {NUM_VALIDATION_IMAGES}"
    return ids


def load(image_id: str) -> tuple[torch.Tensor, np.ndarray]:
    root = dataset_root()
    image = preprocess(SCENE_PX)(Image.open(root / "images/validation" / f"{image_id}.jpg").convert("RGB"))
    annotation = Image.open(root / "annotations/validation" / f"{image_id}.png")
    return image, decode_annotation(preprocess_labels(SCENE_PX)(annotation)[0]).numpy()


def pixels(image: torch.Tensor) -> np.ndarray:
    """An ImageNet-normalized [3, H, W] image as [H, W, 3] uint8 RGB."""
    return (imagenet_denormalize(image.cpu()).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def scene_categories() -> dict[str, str]:
    """Image id to scene category (sceneCategories.txt: training and validation)."""
    lines = (dataset_root() / "sceneCategories.txt").read_text().splitlines()
    return dict(line.split() for line in lines if line)
