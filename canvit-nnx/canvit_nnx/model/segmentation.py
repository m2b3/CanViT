"""CanViT with a linear segmentation probe on the canvas."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from canvit_nnx.model.canvit import CanViT, RecurrentState
from canvit_nnx.model.pretraining import CanViTForPretraining
from canvit_nnx.probes import SegmentationProbe
from canvit_nnx.viewpoint import Viewpoint

Array = jax.Array


def _resize_bilinear(logits: Array, target_size: tuple[int, int]) -> Array:
    assert logits.ndim == 4, logits.shape
    target_height, target_width = target_size
    assert target_height > 0 and target_width > 0, target_size
    batch_size, _, _, channels = logits.shape
    return jax.image.resize(
        logits,
        shape=(batch_size, target_height, target_width, channels),
        method="linear",
        antialias=False,
    )


class CanViTForSemanticSegmentation(nnx.Module):
    def __init__(self, *, canvit: CanViT, probe: SegmentationProbe) -> None:
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
        glimpse: Array,
        state: RecurrentState,
        viewpoint: Viewpoint,
        rngs: jax.Array | nnx.Rngs | None = None,
    ) -> tuple[Array, RecurrentState]:
        """Process a glimpse and return [B, G, G, num_classes] logits with the next state."""
        output = self.canvit(glimpse=glimpse, state=state, viewpoint=viewpoint)
        return self.logits(output.state.canvas, rngs=rngs), output.state

    def logits(self, canvas: Array, *, rngs: jax.Array | nnx.Rngs | None = None) -> Array:
        """Decode a canvas into float32 [B, G, G, num_classes] logits."""
        return self.probe(self.canvit.canvas_patch_grid(canvas).astype(jnp.float32), rngs=rngs)

    def predict(
        self,
        *,
        glimpse: Array,
        state: RecurrentState,
        viewpoint: Viewpoint,
        target_size: tuple[int, int],
        rngs: jax.Array | nnx.Rngs | None = None,
    ) -> tuple[Array, RecurrentState]:
        """Run one glimpse and bilinearly upsample logits to [target_height, target_width]."""
        logits, state = self(glimpse=glimpse, state=state, viewpoint=viewpoint, rngs=rngs)
        return _resize_bilinear(logits, target_size), state


__all__ = ["CanViTForSemanticSegmentation"]
