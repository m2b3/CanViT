"""Viewing policies: generate viewpoint sequences for CanViT.

Pure viewpoint generators — no model or probe dependencies.
Used by pretraining, evaluation, and demos.

Convention: centers are (y, x) in [-1, 1], y downward. Scales in (0, 1].
"""

import torch
from canvit_pytorch.viewpoint import Viewpoint


def random_viewpoints(
    batch_size: int,
    device: torch.device,
    n_viewpoints: int,
    *,
    min_scale: float,
    max_scale: float,
    start_with_full_scene: bool,
) -> list[Viewpoint]:
    """Random viewpoints with safe-box-area scale distribution: p(s) ~ (1-s)."""
    result: list[Viewpoint] = []
    if start_with_full_scene:
        result.append(Viewpoint.full_scene(batch_size=batch_size, device=device))
        n_viewpoints -= 1

    L_min = 1 - max_scale
    L_max = 1 - min_scale
    for _ in range(n_viewpoints):
        u = torch.rand(batch_size, device=device)
        L = torch.sqrt(L_min**2 + u * (L_max**2 - L_min**2))
        scales = 1 - L
        centers = (torch.rand(batch_size, 2, device=device) * 2 - 1) * L.unsqueeze(1)
        result.append(Viewpoint(centers=centers.float(), scales=scales.float()))
    return result


def level_viewpoints(level: int) -> list[tuple[float, float, float]]:
    """(y, x, scale) for all crops at a C2F quadtree level."""
    n = 2**level
    scale = 1.0 / n
    return [
        ((2 * row + 1) * scale - 1.0, (2 * col + 1) * scale - 1.0, scale)
        for row in range(n) for col in range(n)
    ]


def coarse_to_fine_viewpoints(
    batch_size: int,
    device: torch.device,
    n_viewpoints: int,
) -> list[Viewpoint]:
    """Quadtree: full scene -> quadrants -> sub-quadrants. Within-level order shuffled."""
    assert n_viewpoints >= 1
    levels: list[list[tuple[float, float, float]]] = [[(0.0, 0.0, 1.0)]]
    while sum(len(lvl) for lvl in levels) < n_viewpoints:
        parent = levels[-1]
        children: list[tuple[float, float, float]] = []
        for cy, cx, s in parent:
            cs = s / 2
            for qy, qx in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                children.append((cy + (qy - 0.5) * s, cx + (qx - 0.5) * s, cs))
        levels.append(children)
    return _shuffle_levels(levels, batch_size, device, n_viewpoints)


def fine_to_coarse_viewpoints(
    batch_size: int,
    device: torch.device,
    n_viewpoints: int,
) -> list[Viewpoint]:
    """Reversed quadtree: finest scale first, coarsest last."""
    levels: list[list[tuple[float, float, float]]] = []
    total, lvl = 0, 0
    while total < n_viewpoints:
        levels.append(level_viewpoints(lvl))
        total += len(levels[-1])
        lvl += 1
    levels.reverse()
    return _shuffle_levels(levels, batch_size, device, n_viewpoints)


def repeated_full_scene(
    batch_size: int,
    device: torch.device,
    n_viewpoints: int,
) -> list[Viewpoint]:
    """Same full-scene viewpoint at every timestep (recurrence-only control)."""
    return [Viewpoint.full_scene(batch_size=batch_size, device=device)] * n_viewpoints


def _shuffle_levels(
    levels: list[list[tuple[float, float, float]]],
    batch_size: int,
    device: torch.device,
    n_viewpoints: int,
) -> list[Viewpoint]:
    """Iterate levels, shuffle within each, produce viewpoint list."""
    result: list[Viewpoint] = []
    for level_vps in levels:
        t = torch.tensor(level_vps, device=device, dtype=torch.float32)
        n = len(level_vps)
        perms = (torch.zeros(batch_size, 1, dtype=torch.long, device=device) if n == 1
                 else torch.stack([torch.randperm(n, device=device) for _ in range(batch_size)]))
        for i in range(n):
            if len(result) >= n_viewpoints:
                return result
            idx = perms[:, i]
            result.append(Viewpoint(centers=t[idx, :2], scales=t[idx, 2]))
    return result[:n_viewpoints]
