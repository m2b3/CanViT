from typing import Any

import numpy as np
import pytest
import torch
from canvit_pytorch import Viewpoint as TorchViewpoint
from canvit_pytorch import sample_at_viewpoint as torch_sample_at_viewpoint

from tests.backends.support import Backend


@pytest.mark.parametrize(
    ("centers", "scales"),
    [
        ([[0.0, 0.0], [0.1, -0.2]], [0.7, 0.45]),
        ([[0.95, 0.0], [-0.8, 0.25]], [0.6, 0.55]),
        ([[2.0, 0.0], [0.0, -2.0]], [0.2, 0.2]),
    ],
)
def test_sample_at_viewpoint_matches_pytorch_on_rectangular_source(
    backend: Backend, centers: Any, scales: Any,
):
    generator = torch.Generator().manual_seed(1729)
    source = torch.randn((2, 3, 11, 17), generator=generator, dtype=torch.float32)
    centers = np.asarray(centers, dtype=np.float32)
    scales = np.asarray(scales, dtype=np.float32)
    torch_viewpoint = TorchViewpoint(torch.from_numpy(centers), torch.from_numpy(scales))
    torch_result = torch_sample_at_viewpoint(
        spatial=source, viewpoint=torch_viewpoint, glimpse_size_px=7,
    ).permute(0, 2, 3, 1).numpy()
    native_viewpoint = backend.viewpoint(centers, scales)
    native_result = backend.module.sample_at_viewpoint(
        spatial=backend.array(source.permute(0, 2, 3, 1).numpy()),
        viewpoint=native_viewpoint,
        glimpse_size_px=7,
    )
    np.testing.assert_allclose(backend.to_numpy(native_result), torch_result, atol=4e-6, rtol=4e-6)


def test_sample_at_viewpoint_zero_pads_fully_outside_crop(backend: Backend):
    source = torch.ones((2, 3, 9, 13), dtype=torch.float32)
    centers = np.asarray([[2.0, 0.0], [0.0, -2.0]], dtype=np.float32)
    scales = np.asarray([0.2, 0.2], dtype=np.float32)
    native_result = backend.module.sample_at_viewpoint(
        spatial=backend.array(source.permute(0, 2, 3, 1).numpy()),
        viewpoint=backend.viewpoint(centers, scales),
        glimpse_size_px=5,
    )
    np.testing.assert_array_equal(backend.to_numpy(native_result), np.zeros((2, 5, 5, 3), dtype=np.float32))
