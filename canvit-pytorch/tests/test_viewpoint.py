import pytest
import torch

from canvit_pytorch import Viewpoint, sample_at_viewpoint
from canvit_pytorch.viewpoint import crop_box_px


def test_the_full_scene_at_its_own_resolution_is_the_scene():
    scene = torch.randn(2, 3, 32, 32)
    glimpse = sample_at_viewpoint(spatial=scene, viewpoint=Viewpoint.full_scene(batch_size=2, device=None), glimpse_size_px=32)
    torch.testing.assert_close(glimpse, scene)


# Asymmetric crops: swapping rows and columns, or an off-by-half-pixel grid, changes the result.
@pytest.mark.parametrize(("row", "col", "scale"), [(-0.5, -0.5, 0.5), (0.5, -0.25, 0.25), (-0.25, 0.5, 0.5)])
def test_a_pixel_aligned_crop_is_the_image_slice_at_its_box(row, col, scale):
    size = 32
    scene = torch.randn(1, 3, size, size)
    viewpoint = Viewpoint(centers=torch.tensor([[row, col]]), scales=torch.tensor([scale]))
    top, left, bottom, right = crop_box_px(viewpoint, image_size_px=size)[0].round().int().tolist()
    glimpse = sample_at_viewpoint(spatial=scene, viewpoint=viewpoint, glimpse_size_px=bottom - top)
    torch.testing.assert_close(glimpse, scene[:, :, top:bottom, left:right])
