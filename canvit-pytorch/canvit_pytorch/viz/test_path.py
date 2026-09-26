import numpy as np

from canvit_pytorch.policies.random import MIN_SCALE
from canvit_pytorch.viz.path import closed_bezier, point_on, sample_path

# Keypoints at the edges of the valid set, where unconstrained handles would leave it.
KEYPOINTS = [(-0.7, 0.25, 0.2), (0.0, 0.0, 1.0), (0.75, -0.7, 0.25), (0.3, 0.8, 0.18)]


def test_curve_passes_through_keypoints_and_closes():
    segments = closed_bezier(KEYPOINTS)
    assert len(segments) == len(KEYPOINTS)
    for i, keypoint in enumerate(KEYPOINTS):
        np.testing.assert_allclose(point_on(segments, i / len(segments)), keypoint, atol=1e-12)
    np.testing.assert_allclose(segments[-1][-1], KEYPOINTS[0])


def test_every_sample_is_a_crop_inside_the_scene():
    for viewpoint in sample_path(closed_bezier(KEYPOINTS), num_samples=500):
        (row, col), scale = viewpoint.centers[0].tolist(), viewpoint.scales[0].item()
        assert MIN_SCALE - 1e-6 <= scale <= 1 + 1e-6
        assert abs(row) + scale <= 1 + 1e-6 and abs(col) + scale <= 1 + 1e-6
