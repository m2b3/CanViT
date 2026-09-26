"""Viewing policies: where CanViT looks next.

Static generators (coarse-to-fine, fine-to-coarse, random, repeated full scene)
return whole viewpoint sequences; `EntropyGuidedC2F` chooses each tile from the
segmentation probe's uncertainty on the current canvas. `make_policy` builds any
of them by name, as the paper's evaluations and the demos do.

Convention: centers are (row, col) = (y, x) in [-1, 1], y downward. Scales in (0, 1].
"""

import logging
from collections.abc import Callable
from typing import Literal

import torch
from torch import Tensor

from canvit_pytorch.model.base.impl import RecurrentState
from canvit_pytorch.viewpoint import Viewpoint

log = logging.getLogger(__name__)


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

PolicyName = Literal[
    "coarse_to_fine",
    "fine_to_coarse",
    "random",
    "full_then_random",
    "entropy_coarse_to_fine",
    "repeated_full_scene",
]

# entropy_coarse_to_fine excluded: the IN1k classification head doesn't expose
# a get_spatial_fn for the entropy probe.
IN1K_POLICIES: list[PolicyName] = [
    "coarse_to_fine", "fine_to_coarse", "full_then_random", "random", "repeated_full_scene",
]

GetSpatialFn = Callable[[Tensor], Tensor]


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


class StaticPolicy:
    def __init__(self, name: str, viewpoints: list[Viewpoint]) -> None:
        self.name = name
        self._viewpoints = viewpoints

    def step(self, t: int, state: RecurrentState) -> Viewpoint:
        return self._viewpoints[t]


def _build_tile_masks(
    crop_centers: list[tuple[float, float, float]], canvas_grid: int, device: torch.device,
) -> Tensor:
    G = canvas_grid
    assert is_power_of_two(G), f"canvas_grid must be a power of 2, got {G}"
    coords = torch.linspace(-1 + 1 / G, 1 - 1 / G, G, device=device)
    crops_t = torch.tensor(crop_centers, device=device)
    cy, cx, s = crops_t[:, 0], crops_t[:, 1], crops_t[:, 2]
    row_in = (coords.unsqueeze(0) - cy.unsqueeze(1)).abs() <= s.unsqueeze(1)
    col_in = (coords.unsqueeze(0) - cx.unsqueeze(1)).abs() <= s.unsqueeze(1)
    return row_in.unsqueeze(2) & col_in.unsqueeze(1)


class EntropyGuidedC2F:
    """C2F levels visited in order of decreasing per-tile probe entropy."""

    name = "entropy_coarse_to_fine"

    def __init__(
        self, batch_size: int, device: torch.device, canvas_grid: int,
        *, probe: torch.nn.Module, get_spatial_fn: GetSpatialFn,
    ) -> None:
        self._batch_size = batch_size
        self._device = device
        self._canvas_grid = canvas_grid
        self._probe = probe
        self._get_spatial_fn = get_spatial_fn

        # 3 C2F levels: 1 full-scene + 4 half-quadrants + 16 quarter-tiles = 21 timesteps.
        N_LEVELS = 3
        self._levels = [level_viewpoints(lvl) for lvl in range(N_LEVELS)]
        self._level_starts: list[int] = []
        t = 0
        for lvl in self._levels:
            self._level_starts.append(t)
            t += len(lvl)

        self._tile_masks: list[Tensor | None] = [None]
        for lvl in range(1, N_LEVELS):
            self._tile_masks.append(_build_tile_masks(self._levels[lvl], canvas_grid, device))

        self._visited: list[Tensor | None] = [None for _ in self._levels]

    def _compute_entropy(self, state: RecurrentState) -> Tensor:
        spatial = self._get_spatial_fn(state.canvas)
        B, G = spatial.shape[0], self._canvas_grid
        with torch.autocast(device_type=spatial.device.type, enabled=False):
            logits = self._probe(spatial.view(B, G, G, -1).float())
        log_probs = torch.log_softmax(logits, dim=1)
        return -(log_probs.exp() * log_probs).sum(dim=1)

    def step(self, t: int, state: RecurrentState) -> Viewpoint:
        level_idx = sum(1 for s in self._level_starts[1:] if t >= s)
        pos_in_level = t - self._level_starts[level_idx]
        crops = self._levels[level_idx]
        B = self._batch_size

        if level_idx == 0:
            cy, cx, s = crops[0]
            return Viewpoint(
                centers=torch.tensor([[cy, cx]], device=self._device).expand(B, -1),
                scales=torch.full((B,), s, device=self._device),
            )

        if pos_in_level == 0:
            self._visited[level_idx] = torch.zeros(B, len(crops), dtype=torch.bool, device=self._device)

        entropy = self._compute_entropy(state)
        masks = self._tile_masks[level_idx]
        assert masks is not None
        visited = self._visited[level_idx]
        assert visited is not None

        n_cells = masks.sum(dim=(1, 2)).clamp(min=1).float()
        scores = (entropy.unsqueeze(1) * masks.unsqueeze(0).float()).sum(dim=(2, 3)) / n_cells.unsqueeze(0)
        scores = scores.masked_fill(visited, float("-inf"))
        chosen = scores.argmax(dim=1)
        visited.scatter_(1, chosen.unsqueeze(1), True)

        all_crops = torch.tensor(crops, device=self._device)
        selected = all_crops[chosen]
        return Viewpoint(centers=selected[:, :2], scales=selected[:, 2])


def make_policy(
    name: PolicyName,
    batch_size: int,
    device: torch.device,
    n_viewpoints: int,
    *,
    canvas_grid: int = 32,
    min_scale: float = 0.05,
    max_scale: float = 1.0,
    probe: torch.nn.Module | None = None,
    get_spatial_fn: GetSpatialFn | None = None,
) -> StaticPolicy | EntropyGuidedC2F:
    if name == "coarse_to_fine":
        return StaticPolicy(name, coarse_to_fine_viewpoints(batch_size, device, n_viewpoints))
    if name == "fine_to_coarse":
        return StaticPolicy(name, fine_to_coarse_viewpoints(batch_size, device, n_viewpoints))
    if name == "full_then_random":
        return StaticPolicy(name, random_viewpoints(
            batch_size, device, n_viewpoints, min_scale=min_scale, max_scale=max_scale, start_with_full_scene=True,
        ))
    if name == "random":
        return StaticPolicy(name, random_viewpoints(
            batch_size, device, n_viewpoints, min_scale=min_scale, max_scale=max_scale, start_with_full_scene=False,
        ))
    if name == "repeated_full_scene":
        return StaticPolicy(name, repeated_full_scene(batch_size, device, n_viewpoints))
    if name == "entropy_coarse_to_fine":
        assert probe is not None and get_spatial_fn is not None, \
            "entropy_coarse_to_fine requires probe= and get_spatial_fn="
        return EntropyGuidedC2F(batch_size, device, canvas_grid, probe=probe, get_spatial_fn=get_spatial_fn)
    raise ValueError(f"Unknown policy: {name!r}")
