"""Random viewpoints, as sampled in pretraining (paper, Section 5.2) and by the R-IID and F-IID policies."""

import torch

from canvit_pytorch.viewpoint import Viewpoint

MIN_SCALE = 0.05
"""Smallest viewpoint scale: a glimpse covering 0.25% of the scene."""


def random_viewpoints(*, batch_size: int, device: torch.device, min_scale: float = MIN_SCALE) -> Viewpoint:
    """One random viewpoint per scene, scale density p(s) ∝ 1 - s on [min_scale, 1].

    Draws A ~ U([0, (1 - min_scale)²]) and sets s = 1 - √A; the crops of scale s
    that fit in the scene have centers in a box of half side √A, where the
    center is drawn uniformly.
    """
    center_box_half_side = torch.sqrt(torch.rand(batch_size, device=device) * (1 - min_scale) ** 2)
    centers = (torch.rand(batch_size, 2, device=device) * 2 - 1) * center_box_half_side.unsqueeze(1)
    return Viewpoint(centers=centers, scales=1 - center_box_half_side)
