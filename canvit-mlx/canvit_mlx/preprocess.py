"""Pillow and NumPy preprocessing for NHWC MLX inputs."""

import mlx.core as mx
from canvit_core.preprocess import IMAGENET_MEAN, IMAGENET_STD, preprocess, preprocess_labels


def imagenet_normalize(image: mx.array) -> mx.array:
    assert image.ndim >= 3 and image.shape[-1] == 3, image.shape
    mean = mx.array(IMAGENET_MEAN, dtype=image.dtype)
    std = mx.array(IMAGENET_STD, dtype=image.dtype)
    return (image - mean) / std


def imagenet_denormalize(image: mx.array) -> mx.array:
    assert image.ndim >= 3 and image.shape[-1] == 3, image.shape
    mean = mx.array(IMAGENET_MEAN, dtype=image.dtype)
    std = mx.array(IMAGENET_STD, dtype=image.dtype)
    return mx.clip(image * std + mean, 0.0, 1.0)


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "imagenet_denormalize",
    "imagenet_normalize",
    "preprocess",
    "preprocess_labels",
]
