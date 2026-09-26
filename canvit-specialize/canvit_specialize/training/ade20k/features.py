"""Canvas feature extraction for ADE20K probes."""

from dataclasses import dataclass, field

from canvit_pytorch import CanViTForPretrainingHFHub, Viewpoint, sample_at_viewpoint
from torch import Tensor

from canvit_specialize.training.ade20k.config import CanvasFeatureType


@dataclass
class CanvasFeatures:
    """Features from CanViT rollout: one list per feature type, indexed by timestep."""

    features: dict[CanvasFeatureType, list[Tensor]] = field(default_factory=dict)

    def get(self, feat: CanvasFeatureType, t: int) -> Tensor:
        return self.features[feat][t]


def extract_canvas_features(
    *,
    model: CanViTForPretrainingHFHub,
    images: Tensor,
    canvas_grid: int,
    glimpse_px: int,
    viewpoints: list[Viewpoint],
) -> CanvasFeatures:
    """Run CanViT rollout, collect canvas_hidden and recon_normalized at each timestep."""
    B = images.shape[0]
    hidden_list: list[Tensor] = []
    predicted_list: list[Tensor] = []

    state = model.init_state(batch_size=B, canvas_grid_size=canvas_grid)
    for vp in viewpoints:
        glimpse = sample_at_viewpoint(spatial=images, viewpoint=vp, glimpse_size_px=glimpse_px)
        out = model(glimpse=glimpse, state=state, viewpoint=vp)
        state = out.state
        hidden_list.append(model.get_spatial(state.canvas).view(B, canvas_grid, canvas_grid, -1))
        predicted_list.append(model.predict_teacher_scene(state.canvas).view(B, canvas_grid, canvas_grid, -1))

    return CanvasFeatures(features={
        "canvas_hidden": hidden_list,
        "recon_normalized": predicted_list,
    })
