"""ImageNet-1k classification, the paper's global benchmark."""

import os
from pathlib import Path

from torchvision.datasets import ImageFolder
from torchvision.models import ResNet50_Weights

from canvit_pytorch.preprocess import preprocess

CLASS_NAMES: tuple[str, ...] = tuple(ResNet50_Weights.IMAGENET1K_V1.meta["categories"])
NUM_CLASSES = len(CLASS_NAMES)
NUM_TRAIN_IMAGES = 1_281_167
NUM_VALIDATION_IMAGES = 50_000

assert NUM_CLASSES == 1000


def validation_dir() -> Path:
    """$IMAGENET_VAL, the validation images in one subdirectory per class."""
    path = os.environ.get("IMAGENET_VAL")
    assert path, "Set IMAGENET_VAL to the ImageNet-1k validation directory (one subdirectory per class)"
    assert Path(path).is_dir(), f"IMAGENET_VAL={path} is not a directory"
    return Path(path)


def validation_set(directory: Path, *, size_px: int) -> ImageFolder:
    """The paper's evaluation preprocessing: resize the short side to size_px, center-crop."""
    dataset = ImageFolder(str(directory), transform=preprocess(size_px))
    assert len(dataset.classes) == NUM_CLASSES, f"{directory} has {len(dataset.classes)} classes"
    return dataset
