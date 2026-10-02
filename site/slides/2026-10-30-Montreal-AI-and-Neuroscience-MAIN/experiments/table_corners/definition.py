"""Two glimpses at the two ends of a large object, never its middle: does CanViT's canvas predict the object in the
unseen middle? Conditions per example: A (a glimpse at one end), B (at the other end), AB (A, then B). The passive
comparison is DINOv3 ViT-B on each glimpse alone, decoded by its ADE20K probe trained at the glimpse's resolution."""

from dataclasses import dataclass

import numpy as np
import torch
from canvit_pytorch.episode import run_episode
from canvit_pytorch.policies import FixedSequence
from canvit_pytorch.viewpoint import sample_at_viewpoint
from canvit_pytorch.viz.released_model import CANVAS_GRID_SIZE

from experiments.ade20k import SCENE_PX
from experiments.glimpses import GLIMPSE_PX, ViewpointTuple, box_px, viewpoints


@dataclass
class Example:
    image_id: str
    cls: int
    name: str
    horizontal: bool
    a: ViewpointTuple
    b: ViewpointTuple
    object_area: float
    middle_area: float
    mid_a: float = 0.0
    """Of the unseen middle's pixels, the share CanViT labels as the object after A."""
    mid_b: float = 0.0
    mid_ab: float = 0.0
    fp_ab: float = 0.0
    """Of the other labeled pixels between the glimpses, the share labeled as the object after AB."""


def regions(ex: Example, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(middle, corridor): the object's pixels strictly between the two glimpses and outside both; the band between
    the glimpses across the object's extent, for false positives."""
    seen = np.zeros_like(labels, dtype=bool)
    for vp in (ex.a, ex.b):
        top, left, bottom, right = box_px(vp, SCENE_PX)
        seen[max(top, 0):bottom, max(left, 0):right] = True
    ta, la, ba, ra = box_px(ex.a, SCENE_PX)
    tb, lb, bb, rb = box_px(ex.b, SCENE_PX)
    band = np.zeros_like(seen)
    rows, cols = np.nonzero(labels == ex.cls)
    if ex.horizontal:
        lo, hi = (ra, lb) if la < lb else (rb, la)
        band[rows.min():rows.max() + 1, lo:hi] = True
    else:
        lo, hi = (ba, tb) if ta < tb else (bb, ta)
        band[lo:hi, cols.min():cols.max() + 1] = True
    return (labels == ex.cls) & band & ~seen, band & ~seen


@torch.inference_mode()
def canvit_logits(model, images: torch.Tensor, sequence: list[list[ViewpointTuple]]) -> torch.Tensor:
    """sequence[t][i]: example i's viewpoint at glimpse t -> [B, C, G, G] logits decoded after the last glimpse."""
    policy = FixedSequence([viewpoints(step, images.device) for step in sequence])
    steps = run_episode(canvit=model.canvit, images=images, policy=policy, num_glimpses=len(sequence),
                        glimpse_size_px=GLIMPSE_PX,
                        initial_state=model.canvit.init_state(batch_size=images.shape[0], canvas_grid_size=CANVAS_GRID_SIZE))
    return model.logits(steps[-1].state.canvas).float()


@torch.inference_mode()
def dinov3_glimpse_logits(teacher, probe, image: torch.Tensor, vp: ViewpointTuple) -> torch.Tensor:
    """DINOv3 on one glimpse alone, decoded by its ADE20K probe trained at the glimpse's resolution: [C, g, g]."""
    device = next(teacher.parameters()).device
    crop = sample_at_viewpoint(spatial=image[None].to(device), viewpoint=viewpoints([vp], device), glimpse_size_px=GLIMPSE_PX)
    grid = GLIMPSE_PX // teacher.patch_size
    features = teacher(crop).patches.view(1, grid, grid, -1)
    return probe(features.float())[0]
