"""Pillow and NumPy preprocessing shared by the native CanViT ports."""

from collections.abc import Callable

import numpy as np
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_MEAN = np.asarray(IMAGENET_MEAN, dtype=np.float32)
_STD = np.asarray(IMAGENET_STD, dtype=np.float32)


def imagenet_normalize(image: np.ndarray) -> np.ndarray:
    """Normalize an NHWC or HWC float image whose channels are in [0, 1]."""
    image = np.asarray(image)
    assert image.ndim >= 3 and image.shape[-1] == 3, image.shape
    return (image - _MEAN) / _STD


def imagenet_denormalize(image: np.ndarray) -> np.ndarray:
    """Invert ImageNet normalization and clamp channels to [0, 1]."""
    image = np.asarray(image)
    assert image.ndim >= 3 and image.shape[-1] == 3, image.shape
    return np.clip(image * _STD + _MEAN, 0.0, 1.0)


def _resized_short_side(image: Image.Image, size: int, *, interpolation: Image.Resampling) -> Image.Image:
    height, width = image.height, image.width
    short_side, long_side = (width, height) if width <= height else (height, width)
    new_short_side = size
    new_long_side = int(size * long_side / short_side)
    new_width, new_height = (new_short_side, new_long_side) if width <= height else (new_long_side, new_short_side)
    if (new_width, new_height) != image.size:
        image = image.resize((new_width, new_height), resample=interpolation)
    return image


def _center_crop(image: Image.Image, size: int) -> Image.Image:
    top = round((image.height - size) / 2.0)
    left = round((image.width - size) / 2.0)
    return image.crop((left, top, left + size, top + size))


def _validate_size(size: int) -> None:
    assert isinstance(size, int) and size > 0, size


def preprocess(size: int) -> Callable[[Image.Image], np.ndarray]:
    """Return a PIL-to-HWC ImageNet transform matching torchvision's geometry."""
    _validate_size(size)

    def transform(image: Image.Image) -> np.ndarray:
        assert isinstance(image, Image.Image), type(image)
        image = _center_crop(
            _resized_short_side(image, size, interpolation=Image.Resampling.BILINEAR),
            size,
        )
        pixels = np.asarray(image, dtype=np.float32)
        assert pixels.ndim == 3 and pixels.shape[-1] == 3, (
            image.mode,
            pixels.shape,
        )
        return imagenet_normalize(pixels / np.float32(255.0))

    return transform


def preprocess_labels(size: int) -> Callable[[Image.Image], np.ndarray]:
    """Return a PIL-to-HW nearest-neighbor label-map transform."""
    _validate_size(size)

    def transform(image: Image.Image) -> np.ndarray:
        assert isinstance(image, Image.Image), type(image)
        image = _center_crop(
            _resized_short_side(image, size, interpolation=Image.Resampling.NEAREST),
            size,
        )
        labels = np.asarray(image)
        assert labels.ndim == 2, (image.mode, labels.shape)
        return labels.copy()

    return transform


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "imagenet_denormalize",
    "imagenet_normalize",
    "preprocess",
    "preprocess_labels",
]
