"""Smooth viewpoint paths: a closed cubic Bézier curve through keypoint viewpoints, sampled densely.

A viewpoint is (row, col, scale) in scene coordinates. The valid viewpoints
(|row| + scale ≤ 1, |col| + scale ≤ 1, MIN_SCALE ≤ scale ≤ 1) form a convex
set; the curve's control points stay inside it, and a Bézier curve lies in the
convex hull of its control points, so every sampled crop fits in the scene.
"""

import numpy as np
import torch

from canvit_pytorch.policies.random import MIN_SCALE
from canvit_pytorch.viewpoint import Viewpoint

Keypoint = tuple[float, float, float]  # (row, col, scale)
Segment = list[list[float]]  # four control points of one cubic Bézier segment

# Rows of A and b with A @ (row, col, scale) <= b exactly on the valid viewpoints.
_CONSTRAINTS = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1]], dtype=np.float64)
_LIMITS = np.array([1, -MIN_SCALE, 1, 1, 1, 1], dtype=np.float64)


def closed_bezier(keypoints: list[Keypoint]) -> list[Segment]:
    """Segments through the keypoints and back to the first, with a shared tangent at each keypoint."""
    points = np.array(keypoints, dtype=np.float64)
    assert len(points) >= 2 and np.all(_CONSTRAINTS @ points.T <= _LIMITS[:, None] + 1e-9), "a keypoint is not a valid crop"
    points = np.concatenate([points, points[:1]])
    tangents = np.zeros_like(points)
    tangents[1:-1] = (points[2:] - points[:-2]) / 6
    tangents[0] = tangents[-1] = (points[1] - points[-2]) / 6
    for i, point in enumerate(points):  # shorten both handles until they are valid crops
        movement = np.abs(_CONSTRAINTS @ tangents[i])
        slack = np.maximum(0.0, _LIMITS - _CONSTRAINTS @ point)
        moving = movement > 0
        if moving.any():
            tangents[i] *= min(1.0, float(np.min(slack[moving] / movement[moving])))
    return [
        [points[i].tolist(), (points[i] + tangents[i]).tolist(), (points[i + 1] - tangents[i + 1]).tolist(), points[i + 1].tolist()]
        for i in range(len(points) - 1)
    ]


def point_on(segments: list[Segment], phase: float) -> Keypoint:
    """The viewpoint at phase in [0, 1) along the closed curve, each segment taking an equal share."""
    assert 0 <= phase < 1, phase
    position = phase * len(segments)
    index = int(position)
    fraction = position - index
    points = np.array(segments[index])
    while len(points) > 1:  # de Casteljau
        points = points[:-1] + fraction * (points[1:] - points[:-1])
    row, col, scale = points[0].tolist()
    return row, col, scale


def sample_path(segments: list[Segment], num_samples: int) -> list[Viewpoint]:
    """num_samples viewpoints at equal phase steps around the curve, each for a batch of one scene."""
    viewpoints = []
    for i in range(num_samples):
        row, col, scale = point_on(segments, i / num_samples)
        viewpoints.append(Viewpoint(centers=torch.tensor([[row, col]]), scales=torch.tensor([scale])))
    return viewpoints
