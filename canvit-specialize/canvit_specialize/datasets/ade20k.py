"""ADE20K dataset for evaluation and training.

Supports two resize modes:
- "squish": Resize to exact square (default).
- "center_crop": Resize shortest side + CenterCrop.

Results are ONLY comparable across models using the same resize_mode.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

import torch
from PIL import Image
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import Tensor
from torchvision import transforms as T

NUM_CLASSES = 150
IGNORE_LABEL = 255

ResizeMode = Literal["center_crop", "squish"]


class ADE20kDataset(torch.utils.data.Dataset):
    """ADE20K-SceneParse150 dataset.

    Two modes:
    - Separate transforms (eval): img_transform + mask_transform applied independently.
    - Joint transform (training): single callable (img, mask) → (img_t, mask_t).
      Pass joint_transform= and leave img_transform/mask_transform as None.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        img_transform: T.Compose | None = None,
        mask_transform: T.Compose | None = None,
        joint_transform: Callable[[Image.Image, Image.Image], tuple[Tensor, Tensor]] | None = None,
    ) -> None:
        assert (img_transform is not None and mask_transform is not None) or joint_transform is not None, \
            "Provide either (img_transform + mask_transform) or joint_transform"

        img_dir = root / "images" / split
        ann_dir = root / "annotations" / split
        assert img_dir.is_dir(), f"Image dir not found: {img_dir}"
        assert ann_dir.is_dir(), f"Annotation dir not found: {ann_dir}"

        self.images = sorted(img_dir.glob("*.jpg"))
        self.masks = [ann_dir / f"{p.stem}.png" for p in self.images]
        assert len(self.images) > 0, f"No images found in {img_dir}"
        assert all(m.exists() for m in self.masks), "Missing mask files"

        self._img_transform = img_transform
        self._mask_transform = mask_transform
        self._joint_transform = joint_transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        img = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])

        if self._joint_transform is not None:
            img_t, mask_t = self._joint_transform(img, mask)
        else:
            assert self._img_transform is not None and self._mask_transform is not None
            img_t = cast(Tensor, self._img_transform(img))
            # Masks: subtract 1 (ADE20K uses 1-indexed classes, 0 = ignore)
            mask_t = cast(Tensor, self._mask_transform(mask)).squeeze(0).long() - 1
            mask_t[mask_t < 0] = IGNORE_LABEL
        return img_t, mask_t


def make_val_transforms(size: int, mode: ResizeMode) -> tuple[T.Compose, T.Compose]:
    """Create image and mask transforms for ADE20K validation.

    Returns (image_transform, mask_transform).
    """
    if mode == "center_crop":
        img_transform = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(),
                                    T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])
        mask_transform = T.Compose([T.Resize(size, T.InterpolationMode.NEAREST), T.CenterCrop(size),
                                     T.PILToTensor()])
    else:
        img_transform = T.Compose([T.Resize((size, size)), T.ToTensor(),
                                    T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])
        mask_transform = T.Compose([T.Resize((size, size), T.InterpolationMode.NEAREST), T.PILToTensor()])
    return img_transform, mask_transform
