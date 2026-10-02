"""Glimpse sequences of the distillation slide, by name, and which parts of the scene they leave unseen.

A name fixes the viewpoints: `riid-s<seed>` (R-IID, pretraining's random viewpoints after torch.manual_seed(seed)),
`fiid-s<seed>` (the full scene, then random), `c2f-s<seed>` (coarse to fine), `table-ends` (the conference table's two
ends); `-n<count>` after the seed sets the number of glimpses (NUM_GLIMPSES otherwise), the same draws continued. The
sweep gives every scene `riid-s<the number in its id>`. Random draws happen on the given device: the MPS generator
after torch.manual_seed reproduces across processes, the CPU one draws different numbers (the talk's data were drawn
on MPS).
"""

import re

import numpy as np
import torch
from canvit_pytorch.hub import repos
from canvit_pytorch.policies.quadtree import coarse_to_fine
from canvit_pytorch.policies.random import random_viewpoints
from canvit_pytorch.viewpoint import Viewpoint

NUM_GLIMPSES = 8
SCENE_PX = repos.RELEASED_SCENE_SIZE_PX
GLIMPSE_PX = repos.RELEASED_GLIMPSE_SIZE_PX
GRID = repos.RELEASED_CANVAS_GRID_SIZE  # the pretraining readout predicts the teacher's patch grid, one patch per 16 px
PATCH_PX = SCENE_PX // GRID
# The table-corners example: a glimpse at each end of the conference table, never its middle.
TABLE_ENDS = {"ADE_val_00001271": [(-0.078125, -0.0546875, 0.21328125), (0.34765625, 0.78671875, 0.21328125)]}


def sweep_sequence_name(image_id: str) -> str:
    match = re.fullmatch(r"ADE_val_(\d{8})", image_id)
    assert match, f"not an ADE20K validation id: {image_id}"
    return f"riid-s{int(match.group(1))}"


def viewpoint_sequence(name: str, image_id: str, device: torch.device) -> list[Viewpoint]:
    if name == "table-ends":
        return [Viewpoint(centers=torch.tensor([vp[:2]], device=device), scales=torch.tensor([vp[2]], device=device))
                for vp in TABLE_ENDS[image_id]]
    match = re.fullmatch(r"(riid|fiid|c2f)-s(\d+)(?:-n(\d+))?", name)
    assert match, f"unknown sequence name: {name}"
    kind, seed = match.group(1), int(match.group(2))
    count = int(match.group(3)) if match.group(3) else NUM_GLIMPSES
    torch.manual_seed(seed)
    if kind == "c2f":
        return coarse_to_fine(batch_size=1, device=device, num_glimpses=count)
    first = [Viewpoint.full_scene(batch_size=1, device=device)] if kind == "fiid" else []
    return first + [random_viewpoints(batch_size=1, device=device) for _ in range(count - len(first))]


def as_array(viewpoints: list[Viewpoint]) -> np.ndarray:
    """[T, 3] float32 (row, col, scale) of batch-1 viewpoints."""
    assert all(vp.centers.shape[0] == 1 for vp in viewpoints)
    return np.array([[*vp.centers[0].tolist(), float(vp.scales[0])] for vp in viewpoints], dtype=np.float32)


def seen_pixels(viewpoints: np.ndarray) -> np.ndarray:
    """[T, SCENE_PX, SCENE_PX] bool: pixels inside any of glimpses 0..t, each box grown by a pixel because bilinear
    resampling reads the pixels within one pixel of its sample points."""
    centers = np.arange(SCENE_PX) + 0.5
    seen = np.zeros((SCENE_PX, SCENE_PX), dtype=bool)
    out = []
    for row, col, s in viewpoints:
        top, left, bottom, right = ((row - s + 1) * SCENE_PX / 2, (col - s + 1) * SCENE_PX / 2,
                                    (row + s + 1) * SCENE_PX / 2, (col + s + 1) * SCENE_PX / 2)
        rows = (centers > top - 1) & (centers < bottom + 1)
        cols = (centers > left - 1) & (centers < right + 1)
        seen = seen | (rows[:, None] & cols[None, :])
        out.append(seen)
    return np.stack(out)


def never_seen_patches(viewpoints: np.ndarray) -> np.ndarray:
    """[GRID * GRID] bool, row-major like the teacher's patches: no pixel of the patch inside any glimpse."""
    seen = seen_pixels(viewpoints)[-1]
    return ~seen.reshape(GRID, PATCH_PX, GRID, PATCH_PX).any(axis=(1, 3)).reshape(-1)
