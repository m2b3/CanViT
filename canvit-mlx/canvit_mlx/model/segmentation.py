"""CanViT with a linear segmentation probe on the canvas."""

import mlx.core as mx
from mlx import nn

from canvit_mlx.model.canvit import CanViT, RecurrentState
from canvit_mlx.model.pretraining import CanViTForPretraining
from canvit_mlx.module import Module
from canvit_mlx.probes import SegmentationProbe
from canvit_mlx.viewpoint import Viewpoint


def _resize_bilinear(logits: mx.array, target_size: tuple[int, int]) -> mx.array:
    assert logits.ndim == 4, logits.shape
    target_height, target_width = target_size
    assert target_height > 0 and target_width > 0, target_size
    _, height, width, _ = logits.shape
    return nn.Upsample(
        scale_factor=(target_height / height, target_width / width),
        mode="linear",
        align_corners=False,
    )(logits)


class CanViTForSemanticSegmentation(Module):
    def __init__(self, *, canvit: CanViT, probe: SegmentationProbe) -> None:
        super().__init__()
        assert probe.embed_dim == canvit.canvas_dim, (
            f"probe expects {probe.embed_dim}-dim features, canvas has {canvit.canvas_dim}"
        )
        self.canvit = canvit
        self.probe = probe

    @classmethod
    def from_pretrained_with_probe(cls, *, pretrained_repo: str, probe_repo: str) -> "CanViTForSemanticSegmentation":
        """Load a pretrained CanViT and a native probe trained on its canvas."""
        pretrained = CanViTForPretraining.from_pretrained(pretrained_repo)
        return cls(
            canvit=pretrained.canvit,
            probe=SegmentationProbe.from_pretrained(probe_repo),
        )

    @property
    def num_classes(self) -> int:
        return self.probe.num_classes

    def init_state(self, *, batch_size: int, canvas_grid_size: int) -> RecurrentState:
        return self.canvit.init_state(batch_size=batch_size, canvas_grid_size=canvas_grid_size)

    def __call__(
        self,
        *,
        glimpse: mx.array,
        state: RecurrentState,
        viewpoint: Viewpoint,
    ) -> tuple[mx.array, RecurrentState]:
        """Process a glimpse and return [B, G, G, num_classes] logits with the next state."""
        output = self.canvit(glimpse=glimpse, state=state, viewpoint=viewpoint)
        return self.logits(output.state.canvas), output.state

    def logits(self, canvas: mx.array) -> mx.array:
        """Decode a canvas into float32 [B, G, G, num_classes] logits."""
        return self.probe(self.canvit.canvas_patch_grid(canvas).astype(mx.float32))

    def predict(
        self,
        *,
        glimpse: mx.array,
        state: RecurrentState,
        viewpoint: Viewpoint,
        target_size: tuple[int, int],
    ) -> tuple[mx.array, RecurrentState]:
        """Run one glimpse and bilinearly upsample logits to [target_height, target_width]."""
        logits, state = self(glimpse=glimpse, state=state, viewpoint=viewpoint)
        return _resize_bilinear(logits, target_size), state


__all__ = ["CanViTForSemanticSegmentation"]
