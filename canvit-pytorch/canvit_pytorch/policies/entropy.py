"""Entropy-Guided Coarse-to-Fine (EG-C2F): look next where the segmentation is least certain."""

from collections.abc import Callable

import torch
from torch import Tensor

from canvit_pytorch.model.canvit import RecurrentState
from canvit_pytorch.policies.quadtree import Tile, levels_covering, quadtree_level
from canvit_pytorch.viewpoint import Viewpoint, grid_coords

CanvasLogits = Callable[[Tensor], Tensor]
"""Canvas [B, n_tokens, canvas_dim] -> segmentation logits [B, num_classes, G, G]."""


def tile_masks(tiles: list[Tile], *, canvas_grid_size: int, device: torch.device) -> Tensor:
    """[n_tiles, G, G] booleans: which canvas cells each tile covers."""
    cell_centers = grid_coords(size=canvas_grid_size, device=device)[:, 0, 0]  # the same for rows and columns
    table = torch.tensor(tiles, device=device)
    row, col, scale = table[:, 0:1], table[:, 1:2], table[:, 2:3]
    in_rows = (cell_centers - row).abs() <= scale
    in_cols = (cell_centers - col).abs() <= scale
    return in_rows.unsqueeze(2) & in_cols.unsqueeze(1)


def predictive_entropy(logits: Tensor) -> Tensor:
    """[B, num_classes, G, G] logits -> [B, G, G] entropy of each cell's class distribution, in nats."""
    log_probs = torch.log_softmax(logits, dim=1)
    return -(log_probs.exp() * log_probs).sum(dim=1)


def canvas_grid_supported(canvas_grid_size: int) -> bool:
    """Tiles align with canvas cells when the canvas grid is a power of two."""
    return canvas_grid_size > 0 and canvas_grid_size & (canvas_grid_size - 1) == 0


class EntropyGuidedC2F:
    """C2F's quadtree levels in order; within a level, the unvisited tile of highest mean predictive entropy.

    Entropy comes from decoding the current canvas with a segmentation probe,
    recomputed after every glimpse. Deterministic.
    """

    def __init__(
        self, *, batch_size: int, device: torch.device, num_glimpses: int, canvas_grid_size: int,
        canvas_logits: CanvasLogits,
    ) -> None:
        assert canvas_grid_supported(canvas_grid_size), f"EG-C2F needs a power-of-two canvas grid, got {canvas_grid_size}"
        self.batch_size = batch_size
        self.device = device
        self.canvas_logits = canvas_logits
        self.levels = [quadtree_level(level) for level in range(levels_covering(num_glimpses))]
        self.level_start = [sum(len(tiles) for tiles in self.levels[:i]) for i in range(len(self.levels))]
        self.masks = [tile_masks(tiles, canvas_grid_size=canvas_grid_size, device=device) for tiles in self.levels]
        self.visited: Tensor | None = None  # [B, n_tiles] for the level being visited

    def step(self, t: int, state: RecurrentState) -> Viewpoint:
        level = max(i for i, start in enumerate(self.level_start) if t >= start)
        tiles = self.levels[level]
        if level == 0:
            row, col, scale = tiles[0]
            return Viewpoint(
                centers=torch.tensor([[row, col]], device=self.device).expand(self.batch_size, -1),
                scales=torch.full((self.batch_size,), scale, device=self.device),
            )
        if t == self.level_start[level]:
            self.visited = torch.zeros(self.batch_size, len(tiles), dtype=torch.bool, device=self.device)
        assert self.visited is not None
        masks = self.masks[level]
        cells_per_tile = masks.sum(dim=(1, 2)).clamp(min=1).float()
        entropy = predictive_entropy(self.canvas_logits(state.canvas))
        mean_entropy = (entropy.unsqueeze(1) * masks.unsqueeze(0).float()).sum(dim=(2, 3)) / cells_per_tile.unsqueeze(0)
        chosen = mean_entropy.masked_fill(self.visited, float("-inf")).argmax(dim=1)
        self.visited.scatter_(1, chosen.unsqueeze(1), True)
        selected = torch.tensor(tiles, device=self.device)[chosen]
        return Viewpoint(centers=selected[:, :2], scales=selected[:, 2])
