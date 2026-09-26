"""Sample extraction for visualization (shared by train and validation)."""

from dataclasses import dataclass

import numpy as np
from canvit_pytorch import CanViT, CanViTOutput
from torch import Tensor

from .image import imagenet_denormalize_to_numpy


@dataclass
class VizSampleData:
    """Viz data extracted for a single sample.

    Shape annotations:
        G = canvas grid size (e.g., 32)
        g = glimpse grid size (e.g., 8)
        D = teacher feature dim (e.g., 768)
        C = canvas hidden dim
    """

    glimpse: np.ndarray  # [g, g, 3] denormalized RGB
    predicted_scene: np.ndarray  # [G², D] teacher-space prediction
    canvas_spatial: np.ndarray  # [G², C] raw hidden state
    local_patches: np.ndarray | None  # [g², D] local stream patches (None if not available)


def extract_sample0_viz(
    out: CanViTOutput,
    glimpse: Tensor,
    predicted_scene: Tensor,
    model: CanViT,
) -> VizSampleData:
    """Extract viz data for sample 0, move to CPU as numpy."""
    glimpse_cpu = glimpse[0].detach().cpu()
    glimpse_np = imagenet_denormalize_to_numpy(glimpse_cpu)

    scene_cpu = predicted_scene[0].detach().cpu().float()
    scene_np = scene_cpu.numpy()

    canvas_single = out.state.canvas[0:1].detach()
    spatial = model.get_spatial(canvas_single)[0]
    spatial_np = spatial.cpu().float().numpy()

    # Extract local stream patches if available
    local_np: np.ndarray | None = None
    if out.local_patches is not None:
        local_np = out.local_patches[0].detach().cpu().float().numpy()

    return VizSampleData(
        glimpse=glimpse_np,
        predicted_scene=scene_np,
        canvas_spatial=spatial_np,
        local_patches=local_np,
    )
