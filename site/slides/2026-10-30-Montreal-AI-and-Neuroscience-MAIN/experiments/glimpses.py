"""Viewpoints as the experiments write them: (row, col, scale) tuples in scene coordinates, the scale being the crop's
half side; the crops they cover; and what CanViT receives there."""

from collections.abc import Sequence

import numpy as np
import torch
from canvit_pytorch.hub.repos import RELEASED_GLIMPSE_SIZE_PX
from canvit_pytorch.viewpoint import Viewpoint, sample_at_viewpoint

from experiments.ade20k import pixels

GLIMPSE_PX = RELEASED_GLIMPSE_SIZE_PX

ViewpointTuple = tuple[float, float, float]
FULL_SCENE: ViewpointTuple = (0.0, 0.0, 1.0)


def viewpoints(rows: Sequence[ViewpointTuple], device: torch.device) -> Viewpoint:
    """One viewpoint per batch item."""
    t = torch.tensor(rows, dtype=torch.float32, device=device)
    return Viewpoint(centers=t[:, :2].contiguous(), scales=t[:, 2].contiguous())


def box_px(vp: ViewpointTuple | np.ndarray, size: int) -> tuple[int, int, int, int]:
    """(top, left, bottom, right) of a viewpoint's crop in a size-px scene, rounded to the nearest pixel."""
    row, col, s = vp
    return (round((row - s + 1) * size / 2), round((col - s + 1) * size / 2),
            round((row + s + 1) * size / 2), round((col + s + 1) * size / 2))


@torch.inference_mode()
def glimpse_pixels(image: torch.Tensor, vp: ViewpointTuple, device: torch.device) -> np.ndarray:
    """The [g, g, 3] uint8 glimpse CanViT receives at vp: run_episode's sampling of the normalized scene."""
    crop = sample_at_viewpoint(spatial=image[None].to(device), viewpoint=viewpoints([vp], device),
                               glimpse_size_px=GLIMPSE_PX)
    return pixels(crop[0])
