"""The viewpoints a learned policy chooses among."""

from dataclasses import dataclass

import torch
from torch import Tensor

from canvit_pytorch.viewpoint import Viewpoint


@dataclass(frozen=True)
class CandidateViewpoints:
    """At each scale s, a centers_per_axis × centers_per_axis grid of centers spanning [-(1 - s), 1 - s]².

    Every glimpse lies inside the scene, and the grid spacing does not depend on the scale, so glimpses of a
    scale overlap. Candidates are numbered scale by scale, then row by row: the order of ViewpointScorer's scores.
    """

    scales: tuple[float, ...]
    centers_per_axis: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "scales", tuple(self.scales))  # a config.json holds a list
        assert self.scales and all(0 < scale <= 1 for scale in self.scales) and self.centers_per_axis >= 1, self

    def grid(self) -> Tensor:
        """[num_scales, centers_per_axis, centers_per_axis, 3]: each candidate's (row, col, scale), on the CPU."""
        per_scale = []
        for scale in self.scales:
            centers = torch.linspace(-(1 - scale), 1 - scale, self.centers_per_axis)
            rows, cols = torch.meshgrid(centers, centers, indexing="ij")
            per_scale.append(torch.stack([rows, cols, torch.full_like(rows, scale)], dim=-1))
        return torch.stack(per_scale)

    def viewpoints(self, index: Tensor) -> Viewpoint:
        """The candidates numbered index [B]."""
        chosen = self.grid().flatten(0, 2).to(index.device)[index]
        return Viewpoint(centers=chosen[:, :2].contiguous(), scales=chosen[:, 2].contiguous())
