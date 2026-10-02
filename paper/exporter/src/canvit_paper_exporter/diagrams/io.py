"""Image input and output of the diagram generators, and the paper's PCA colors of token grids."""

import math
from pathlib import Path

import numpy as np
import torch
from canvit_pytorch.preprocess import imagenet_denormalize, imagenet_normalize
from canvit_pytorch.viz.pca import PCABasis, color_limits, fit_pca, project, to_rgb
from PIL import Image
from PIL.Image import Resampling
from torch import Tensor
from torchvision.transforms.functional import to_tensor

OUTPUT_SIZE = 512


def load_image(path: Path, device: torch.device) -> tuple[Tensor, Image.Image]:
    """The image center-cropped to a square: ImageNet-normalized [1, 3, H, W], and as PIL."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    return imagenet_normalize(to_tensor(img)).unsqueeze(0).to(device), img


def save_image(img: Image.Image, path: Path, size: int = OUTPUT_SIZE) -> None:
    img.resize((size, size), Resampling.LANCZOS).save(path)


def denormalized_numpy(image: Tensor) -> np.ndarray:
    """ImageNet-normalized [3, H, W] -> [H, W, 3] float32 in [0, 1]."""
    return imagenet_denormalize(image.cpu().float()).permute(1, 2, 0).numpy()


def save_tensor_rgb(tensor: Tensor, path: Path, size: int = OUTPUT_SIZE) -> None:
    """Save an ImageNet-normalized [3, H, W] tensor as an RGB image."""
    arr = (denormalized_numpy(tensor) * 255).astype(np.uint8)
    Image.fromarray(arr).resize((size, size), Resampling.LANCZOS).save(path)


def pca_colors(basis: PCABasis, tokens: np.ndarray, *, shape: tuple[int, int] | None = None) -> np.ndarray:
    """[N, D] tokens -> [H, W, 3] uint8 colors in the basis, min-max scaled over these tokens; square by default."""
    if shape is None:
        side = math.isqrt(len(tokens))
        assert side * side == len(tokens), f"{len(tokens)} tokens do not form a square grid"
        shape = (side, side)
    projection = project(basis, tokens)
    return to_rgb(projection, color_limits(projection)).reshape(*shape, 3)


def save_pca(tokens: Tensor, path: Path, basis: PCABasis | None = None, size: int = OUTPUT_SIZE) -> PCABasis:
    """Save [N, D] tokens as PCA colors, in basis or one fitted on them; return the basis."""
    feats = tokens.detach().cpu().float().numpy()
    basis = fit_pca(feats) if basis is None else basis
    Image.fromarray(pca_colors(basis, feats)).resize((size, size), Resampling.NEAREST).save(path)
    return basis
