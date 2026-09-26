"""Image transformation utilities for visualization."""

from canvit_pytorch.preprocess import imagenet_denormalize
from torch import Tensor


def imagenet_denormalize_to_numpy(img: Tensor):
    """Denormalize ImageNet-normalized tensor and return [H, W, C] numpy in [0, 1].

    Casts to float32: under AMP the tensor may be bfloat16, which numpy cannot convert.
    """
    return imagenet_denormalize(img).detach().cpu().float().permute(1, 2, 0).numpy()


__all__ = ["imagenet_denormalize_to_numpy"]
