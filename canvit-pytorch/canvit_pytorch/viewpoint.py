"""The scene frame: coordinates, viewpoints and glimpse extraction.

Scene coordinates span [-1, 1]² in (row, col) order, like tensor indexing:
(-1, -1) is the top-left corner. A viewpoint is a square crop given by its
center and its scale, the crop's half side, so it covers scale² of the scene.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


def grid_coords(*, size: int, device: torch.device | None) -> Tensor:
    """Cell centers of a uniform size×size grid over [-1, 1]²: [size, size, 2], float32."""
    centers = (torch.arange(size, device=device, dtype=torch.float32) + 0.5) / size * 2 - 1
    return torch.stack(torch.meshgrid(centers, centers, indexing="ij"), dim=-1)


@dataclass(frozen=True)
class Viewpoint:
    centers: Tensor  # [B, 2] float32, (row, col) in [-1, 1]
    scales: Tensor  # [B] float32 in (0, 1]

    def __post_init__(self) -> None:
        assert self.centers.dtype == self.scales.dtype == torch.float32, (self.centers.dtype, self.scales.dtype)
        assert self.centers.ndim == 2 and self.centers.shape == (self.scales.shape[0], 2), (
            self.centers.shape, self.scales.shape,
        )

    @staticmethod
    def full_scene(*, batch_size: int, device: torch.device | None) -> "Viewpoint":
        return Viewpoint(
            centers=torch.zeros(batch_size, 2, device=device, dtype=torch.float32),
            scales=torch.ones(batch_size, device=device, dtype=torch.float32),
        )


def viewpoint_grid_coords(viewpoint: Viewpoint, *, size: int) -> Tensor:
    """Scene coordinates of a size×size grid spanning each viewpoint's crop: [B, size, size, 2]."""
    B = viewpoint.centers.shape[0]
    offsets = grid_coords(size=size, device=viewpoint.centers.device)
    return viewpoint.centers.view(B, 1, 1, 2) + viewpoint.scales.view(B, 1, 1, 1) * offsets


def crop_box_px(viewpoint: Viewpoint, *, image_size_px: int) -> Tensor:
    """[B, 4]: (top, left, bottom, right) of each viewpoint's crop, in pixels from the image's top-left corner."""
    half_side = viewpoint.scales.unsqueeze(1)
    corners = torch.cat([viewpoint.centers - half_side, viewpoint.centers + half_side], dim=1)
    return (corners + 1) * (image_size_px / 2)


def sample_at_viewpoint(*, spatial: Tensor, viewpoint: Viewpoint, glimpse_size_px: int) -> Tensor:
    """Bilinearly resample each viewpoint's crop of [B, C, H, W] to [B, C, glimpse_size_px, glimpse_size_px].

    Works on images and on feature maps alike; computes in float32, returns spatial's dtype.
    """
    xy = viewpoint_grid_coords(viewpoint, size=glimpse_size_px).flip(-1)  # grid_sample takes (x, y)
    return F.grid_sample(spatial.float(), xy, mode="bilinear", align_corners=False).to(spatial.dtype)
