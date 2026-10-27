"""DINOv3's photometric distortion (dinov3/eval/segmentation/transforms.py::PhotoMetricDistortion) on uint8 images.

Each operation applies with probability 1/2, in the reference code's order:
brightness, contrast (before or after the rest, at random), saturation, hue.
Saturation and hue follow torchvision's float formulas.
"""

import numpy as np

BRIGHTNESS_DELTA = 32.0
CONTRAST_RANGE = (0.5, 1.5)
SATURATION_RANGE = (0.5, 1.5)
HUE_RANGE = (-0.5, 0.5)
"""In turns of the hue circle: the whole circle."""
LUMA = np.array([0.2989, 0.587, 0.114], np.float32)
"""ITU-R 601-2 luma, toward which torchvision's adjust_saturation blends."""


def _affine(image: np.ndarray, *, scale: float = 1.0, shift: float = 0.0) -> np.ndarray:
    return np.clip(image.astype(np.float32) * scale + shift, 0, 255).astype(np.uint8)


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """[..., 3] RGB in [0, 1] to hue, saturation and value, each in [0, 1]; gray pixels get hue 0."""
    value = rgb.max(-1)
    chroma = value - rgb.min(-1)
    divisor = np.where(chroma == 0, 1.0, chroma)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    hue = np.where(
        value == red,
        ((green - blue) / divisor) % 6.0,
        np.where(value == green, (blue - red) / divisor + 2.0, (red - green) / divisor + 4.0),
    )
    hue = np.where(chroma == 0, 0.0, hue / 6.0)
    saturation = np.where(value == 0, 0.0, chroma / np.where(value == 0, 1.0, value))
    return np.stack([hue, saturation, value], -1)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sextant = np.floor(hue * 6.0)
    fraction = hue * 6.0 - sextant
    p, q, t = value * (1 - saturation), value * (1 - fraction * saturation), value * (1 - (1 - fraction) * saturation)
    index = (sextant % 6).astype(np.int32)
    red = np.choose(index, [value, q, p, p, t, value])
    green = np.choose(index, [t, value, value, q, p, p])
    blue = np.choose(index, [p, p, t, value, value, q])
    return np.stack([red, green, blue], -1)


def _saturate(image: np.ndarray, factor: float) -> np.ndarray:
    gray = (image.astype(np.float32) * LUMA).sum(-1, keepdims=True)
    return np.clip(factor * image.astype(np.float32) + (1 - factor) * gray, 0, 255).astype(np.uint8)


def _rotate_hue(image: np.ndarray, turns: float) -> np.ndarray:
    hsv = _rgb_to_hsv(image.astype(np.float32) / 255.0)
    hsv[..., 0] = (hsv[..., 0] + turns) % 1.0
    return np.clip(_hsv_to_rgb(hsv) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def photometric_distortion(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Distort an [H, W, 3] uint8 image."""
    assert image.dtype == np.uint8 and image.shape[-1] == 3, (image.dtype, image.shape)
    if rng.integers(2):
        image = _affine(image, shift=float(rng.uniform(-BRIGHTNESS_DELTA, BRIGHTNESS_DELTA)))
    contrast_last = int(rng.integers(2)) == 0

    def contrast(image: np.ndarray) -> np.ndarray:
        return _affine(image, scale=float(rng.uniform(*CONTRAST_RANGE))) if rng.integers(2) else image

    if not contrast_last:
        image = contrast(image)
    if rng.integers(2):
        image = _saturate(image, float(rng.uniform(*SATURATION_RANGE)))
    if rng.integers(2):
        image = _rotate_hue(image, float(rng.uniform(*HUE_RANGE)))
    if contrast_last:
        image = contrast(image)
    return image
