"""Pillow and NumPy preprocessing for NHWC NNX inputs."""

import jax.numpy as jnp
from canvit_core.preprocess import IMAGENET_MEAN, IMAGENET_STD, preprocess, preprocess_labels


def imagenet_normalize(image: jnp.ndarray) -> jnp.ndarray:
    assert image.ndim >= 3 and image.shape[-1] == 3, image.shape
    mean = jnp.asarray(IMAGENET_MEAN, dtype=image.dtype)
    std = jnp.asarray(IMAGENET_STD, dtype=image.dtype)
    return (image - mean) / std


def imagenet_denormalize(image: jnp.ndarray) -> jnp.ndarray:
    assert image.ndim >= 3 and image.shape[-1] == 3, image.shape
    mean = jnp.asarray(IMAGENET_MEAN, dtype=image.dtype)
    std = jnp.asarray(IMAGENET_STD, dtype=image.dtype)
    return jnp.clip(image * std + mean, 0.0, 1.0)


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "imagenet_denormalize",
    "imagenet_normalize",
    "preprocess",
    "preprocess_labels",
]
