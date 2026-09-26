"""The quadtree of viewpoints behind the C2F, F2C and EG-C2F policies (paper, Appendix D.1).

Level ℓ tiles the scene with a 2^ℓ × 2^ℓ grid of non-overlapping crops of scale 2^-ℓ.
"""

from typing import NamedTuple

import torch

from canvit_pytorch.viewpoint import Viewpoint


class Tile(NamedTuple):
    row: float  # center, scene coordinates
    col: float
    scale: float


def quadtree_level(level: int) -> list[Tile]:
    """The level's tiles, in row-major order."""
    n = 2**level
    scale = 1.0 / n
    return [Tile((2 * r + 1) * scale - 1.0, (2 * c + 1) * scale - 1.0, scale) for r in range(n) for c in range(n)]


def quadtree_level_z_order(level: int) -> list[Tile]:
    """The level's tiles in Z-order: each parent tile's four children, parents in Z-order.

    C2F enumerates levels this way; the enumeration fixes which permutation a
    random seed produces, so it stays as it was for the paper's evaluations.
    """
    n = 2**level

    def z_index(tile_index: int) -> int:
        r, c = divmod(tile_index, n)
        return sum((((r >> b) & 1) << (2 * b + 1)) | (((c >> b) & 1) << (2 * b)) for b in range(level))

    tiles = quadtree_level(level)
    return [tiles[i] for i in sorted(range(n * n), key=z_index)]


def levels_covering(num_glimpses: int) -> int:
    """How many quadtree levels (1, 4, 16, ... tiles) it takes to reach num_glimpses viewpoints."""
    levels, total = 0, 0
    while total < num_glimpses:
        total += 4**levels
        levels += 1
    return levels


def shuffled_levels(
    levels: list[list[Tile]], *, batch_size: int, device: torch.device, num_glimpses: int,
) -> list[Viewpoint]:
    """Visit the levels in the given order, each level's tiles in an independent random order per scene."""
    viewpoints: list[Viewpoint] = []
    for tiles in levels:
        table = torch.tensor(tiles, device=device, dtype=torch.float32)
        n = len(tiles)
        orders = (torch.zeros(batch_size, 1, dtype=torch.long, device=device) if n == 1
                  else torch.stack([torch.randperm(n, device=device) for _ in range(batch_size)]))
        for i in range(n):
            if len(viewpoints) == num_glimpses:
                return viewpoints
            chosen = table[orders[:, i]]
            viewpoints.append(Viewpoint(centers=chosen[:, :2], scales=chosen[:, 2]))
    return viewpoints


def coarse_to_fine(*, batch_size: int, device: torch.device, num_glimpses: int) -> list[Viewpoint]:
    levels = [quadtree_level_z_order(level) for level in range(levels_covering(num_glimpses))]
    return shuffled_levels(levels, batch_size=batch_size, device=device, num_glimpses=num_glimpses)


def fine_to_coarse(*, batch_size: int, device: torch.device, num_glimpses: int) -> list[Viewpoint]:
    levels = [quadtree_level(level) for level in reversed(range(levels_covering(num_glimpses)))]
    return shuffled_levels(levels, batch_size=batch_size, device=device, num_glimpses=num_glimpses)
