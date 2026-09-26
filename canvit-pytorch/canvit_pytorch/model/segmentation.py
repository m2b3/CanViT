"""CanViT with a linear segmentation probe on the canvas."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from canvit_pytorch.model.canvit import CanViT, RecurrentState
from canvit_pytorch.model.pretraining import CanViTForPretraining
from canvit_pytorch.probes import SegmentationProbe
from canvit_pytorch.viewpoint import Viewpoint


class CanViTForSemanticSegmentation(nn.Module):
    def __init__(self, *, canvit: CanViT, probe: SegmentationProbe) -> None:
        super().__init__()
        assert probe.embed_dim == canvit.canvas_dim, f"probe expects {probe.embed_dim}-dim features, canvas has {canvit.canvas_dim}"
        self.canvit = canvit
        self.probe = probe

    @classmethod
    def from_pretrained_with_probe(cls, *, pretrained_repo: str, probe_repo: str) -> "CanViTForSemanticSegmentation":
        """Hub repo ids or local directories of a pretrained CanViT and a probe trained on its canvas."""
        pretrained = CanViTForPretraining.from_pretrained(pretrained_repo)
        return cls(canvit=pretrained.canvit, probe=SegmentationProbe.from_pretrained(probe_repo))

    @property
    def num_classes(self) -> int:
        return self.probe.num_classes

    def init_state(self, *, batch_size: int, canvas_grid_size: int) -> RecurrentState:
        return self.canvit.init_state(batch_size=batch_size, canvas_grid_size=canvas_grid_size)

    def forward(self, *, glimpse: Tensor, state: RecurrentState, viewpoint: Viewpoint) -> tuple[Tensor, RecurrentState]:
        """(logits [B, num_classes, G, G] on the canvas grid, next state)."""
        out = self.canvit(glimpse=glimpse, state=state, viewpoint=viewpoint)
        return self.logits(out.state.canvas), out.state

    def logits(self, canvas: Tensor) -> Tensor:
        """Decode a canvas into [B, num_classes, G, G] float32 logits, also under autocast."""
        with torch.autocast(device_type=canvas.device.type, enabled=False):
            return self.probe(self.canvit.canvas_patch_grid(canvas).float())

    def predict(
        self, *, glimpse: Tensor, state: RecurrentState, viewpoint: Viewpoint, target_size: tuple[int, int],
    ) -> tuple[Tensor, RecurrentState]:
        """forward, with logits bilinearly upsampled to target_size."""
        logits, state = self(glimpse=glimpse, state=state, viewpoint=viewpoint)
        return F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False), state
