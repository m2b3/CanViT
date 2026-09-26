"""ADE20K SceneParsing (ADEChallengeData2016): images, label maps, and the evaluation transform."""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from canvit_pytorch.benchmarks.ade20k.classes import IGNORE_LABEL
from canvit_pytorch.preprocess import IMAGENET_MEAN, IMAGENET_STD

Split = Literal["training", "validation"]
Transform = Callable[[Image.Image, Image.Image], tuple[Tensor, Tensor]]
"""(RGB image, label map) -> (normalized [3, H, W], labels [H, W] in 0..149 or IGNORE_LABEL)."""

NUM_VALIDATION_IMAGES = 2000


def decode_annotation(annotation: Tensor) -> Tensor:
    """ADE20K annotation values (0 unlabeled, k the k-th class) -> class indices 0..149, IGNORE_LABEL where unlabeled."""
    labels = annotation.long() - 1
    labels[labels < 0] = IGNORE_LABEL
    return labels


def dataset_root() -> Path:
    """$ADE20K_ROOT, the ADEChallengeData2016 directory."""
    root = os.environ.get("ADE20K_ROOT")
    assert root, "Set ADE20K_ROOT to the ADEChallengeData2016 directory"
    assert Path(root).is_dir(), f"ADE20K_ROOT={root} is not a directory"
    return Path(root)


class EvaluationTransform:
    """Resize to size_px × size_px without cropping, as the paper's active-vision baselines evaluate.

    A class rather than a closure, so DataLoader workers started by spawn (macOS) can unpickle it.
    """

    def __init__(self, size_px: int) -> None:
        self.image_transform = transforms.Compose([
            transforms.Resize((size_px, size_px)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.label_transform = transforms.Compose([
            transforms.Resize((size_px, size_px), transforms.InterpolationMode.NEAREST),
            transforms.PILToTensor(),
        ])

    def __call__(self, image: Image.Image, label_map: Image.Image) -> tuple[Tensor, Tensor]:
        label_tensor = self.label_transform(label_map)
        assert isinstance(label_tensor, Tensor)
        labels = decode_annotation(label_tensor.squeeze(0))
        image_tensor = self.image_transform(image)
        assert isinstance(image_tensor, Tensor)
        return image_tensor, labels


def evaluation_transform(size_px: int) -> Transform:
    return EvaluationTransform(size_px)


class ADE20kDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, *, root: Path, split: Split, transform: Transform) -> None:
        image_dir, label_dir = root / "images" / split, root / "annotations" / split
        self.images = sorted(image_dir.glob("*.jpg"))
        assert self.images, f"no images in {image_dir}"
        self.label_maps = [label_dir / f"{image.stem}.png" for image in self.images]
        missing = [path for path in self.label_maps if not path.exists()]
        assert not missing, f"{len(missing)} label maps missing, e.g. {missing[0]}"
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image = Image.open(self.images[index]).convert("RGB")
        label_map = Image.open(self.label_maps[index])
        image_tensor, labels = self.transform(image, label_map)
        assert labels.dtype == torch.long, labels.dtype
        return image_tensor, labels
