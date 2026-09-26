from dataclasses import dataclass
from enum import Enum, auto
from typing import NamedTuple

import torch
from canvit_pytorch.policies import coarse_to_fine_viewpoints as _coarse_to_fine
from canvit_pytorch.viewpoint import Viewpoint as CoreViewpoint
from torch import Tensor

__all__ = [
    "PixelBox",
    "Viewpoint",
    "ViewpointType",
    "make_eval_viewpoints",
    "viewpoint_to_pixel_box",
]


class ViewpointType(Enum):
    """Type of viewpoint for training branches."""

    RANDOM = auto()
    FULL = auto()


class PixelBox(NamedTuple):
    """Axis-aligned bounding box in pixel coordinates."""

    left: float
    top: float
    width: float
    height: float
    center_x: float
    center_y: float


def viewpoint_to_pixel_box(
    centers: Tensor, scales: Tensor, batch_idx: int, H: int, W: int
) -> PixelBox:
    """Convert viewpoint geometry to pixel coordinates for visualization.

    Maps normalized [-1, 1] to pixel centers [0, W-1] and [0, H-1].
    """
    cy, cx = centers[batch_idx].tolist()
    scale = scales[batch_idx].item()
    # Map normalized [-1, 1] to pixel [0, W-1] (pixel center convention)
    center_x = (cx + 1) / 2 * (W - 1)
    center_y = (cy + 1) / 2 * (H - 1)
    width = scale * (W - 1)
    height = scale * (H - 1)
    return PixelBox(
        left=center_x - width / 2,
        top=center_y - height / 2,
        width=width,
        height=height,
        center_x=center_x,
        center_y=center_y,
    )


@dataclass
class Viewpoint(CoreViewpoint):
    """Viewpoint with name for debugging/logging."""

    name: str = ""

    def to_pixel_box(self, batch_idx: int, H: int, W: int) -> "PixelBox":
        """Convert to pixel coordinates for visualization."""
        return viewpoint_to_pixel_box(self.centers, self.scales, batch_idx, H, W)

    @staticmethod
    def full_scene(*, batch_size: int, device: torch.device) -> "Viewpoint":
        return Viewpoint(
            name="full",
            centers=torch.zeros(batch_size, 2, device=device),
            scales=torch.ones(batch_size, device=device),
        )

    @staticmethod
    def quadrant(B: int, device: torch.device, qx: int, qy: int) -> "Viewpoint":
        """Quadrant viewpoint: qx,qy in {0,1} -> center, scale=0.5."""
        cx = -0.5 + qx
        cy = -0.5 + qy
        name = ["TL", "TR", "BL", "BR"][qy * 2 + qx]
        centers = torch.tensor([[cy, cx]], device=device).expand(B, -1)
        return Viewpoint(
            name=name,
            centers=centers,
            scales=torch.full((B,), 0.5, device=device),
        )

    @staticmethod
    def random(
        *, batch_size: int, device: torch.device, min_scale: float, max_scale: float = 1.0
    ) -> "Viewpoint":
        """Sample random viewpoints with uniform safe-box-area distribution.

        Geometry: viewpoint has center (x, y) ∈ [-1, 1]² and scale s ∈ [min_scale, max_scale].
        Constraint: |x| + s ≤ 1 and |y| + s ≤ 1 (viewpoint must fit in scene).
        Given scale s, valid centers form a "safe box": [-(1-s), (1-s)]² with area A = 4·(1-s)².

        We weight each scale by its safe-box half-side-length (1-s) via the L²-uniform trick,
        yielding p(s) ∝ (1-s). This favors zoomed-in glimpses while accounting for geometry.
        """
        assert 0.0 <= min_scale <= max_scale <= 1.0

        L_min = 1 - max_scale
        L_max = 1 - min_scale

        u = torch.rand(batch_size, device=device)
        L_sq = L_min**2 + u * (L_max**2 - L_min**2)
        L = torch.sqrt(L_sq)

        scales = 1 - L
        centers = (torch.rand(batch_size, 2, device=device) * 2 - 1) * L.unsqueeze(1)

        return Viewpoint(name="random", centers=centers, scales=scales)


def make_eval_viewpoints(
    B: int, device: torch.device, n_viewpoints: int = 10
) -> list[Viewpoint]:
    """Generate quadtree viewpoints with random ordering WITHIN each level, per batch item.

    Wraps canvit_pytorch.policies.coarse_to_fine_viewpoints and adds names for debugging.
    """
    core_vps = _coarse_to_fine(B, device, n_viewpoints)
    result: list[Viewpoint] = []
    for i, vp in enumerate(core_vps):
        name = "full" if i == 0 else f"vp_{i}"
        result.append(Viewpoint(name=name, centers=vp.centers, scales=vp.scales))
    return result
