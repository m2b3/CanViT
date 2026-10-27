"""CanViT with a CenterNet detection head on the canvas."""

import dataclasses
from pathlib import Path
from typing import NamedTuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from canvit_core.config import CanViTConfig

from canvit_nnx.hub import HubMixin
from canvit_nnx.model.canvit import CanViT, RecurrentState
from canvit_nnx.model.config import ComputeDtype
from canvit_nnx.model.pretraining import CanViTForPretraining
from canvit_nnx.viewpoint import Viewpoint

Array = jax.Array

HEATMAP_BIAS = -2.19
"""CenterNet's initial heatmap logit: sigmoid(-2.19) ≈ 0.1 keeps the focal loss's negative term in range at the start."""


class CenterNetMaps(NamedTuple):
    heatmap: Array  # [B, G, G, num_classes] logits of an object center in each cell, per class
    size: Array  # [B, G, G, 2] box (height, width) in cells
    offset: Array  # [B, G, G, 2] center minus the cell's corner, in cells


class CenterNetHead(nnx.Module):
    """A CenterNet head on canvas patches: a shared 3×3 convolution, then 1×1 convolutions to each map.

    CenterNet gives each map its own 3×3 convolution; on the wide canvas a
    3×3 convolution dominates the head's cost, so the maps share one.
    LayerNorm and BatchNorm first, as in the segmentation probe, bring the
    canvas to the scale that HEATMAP_BIAS assumes.
    """

    def __init__(self, *, in_dim: int, num_classes: int, width: int, rngs: nnx.Rngs) -> None:
        self.layer_norm = nnx.LayerNorm(in_dim, rngs=rngs)
        self.batch_norm = nnx.BatchNorm(in_dim, rngs=rngs)
        self.stem = nnx.Conv(in_dim, width, kernel_size=(3, 3), rngs=rngs)
        self.heatmap = nnx.Conv(
            width, num_classes, kernel_size=(1, 1), bias_init=nnx.initializers.constant(HEATMAP_BIAS), rngs=rngs
        )
        self.size = nnx.Conv(width, 2, kernel_size=(1, 1), rngs=rngs)
        self.offset = nnx.Conv(width, 2, kernel_size=(1, 1), rngs=rngs)

    def __call__(self, canvas_patches: Array) -> CenterNetMaps:
        """Map [B, G, G, in_dim] canvas patches to CenterNet's maps over the same grid."""
        features = nnx.relu(self.stem(self.batch_norm(self.layer_norm(canvas_patches))))
        return CenterNetMaps(heatmap=self.heatmap(features), size=self.size(features), offset=self.offset(features))


class CanViTForObjectDetection(nnx.Module, HubMixin):
    """CanViT and a CenterNet head that reads the canvas after every glimpse.

    The head's BatchNorm normalizes with batch statistics; evaluate a view with
    use_running_average=True (nnx.view).
    """

    def __init__(
        self,
        *,
        canvit_config: CanViTConfig,
        num_classes: int,
        head_width: int,
        rngs: nnx.Rngs,
        compute_dtype: ComputeDtype = "float32",
    ) -> None:
        self.canvit = CanViT(canvit_config, rngs=rngs, compute_dtype=compute_dtype)
        self.head = CenterNetHead(in_dim=self.canvit.canvas_dim, num_classes=num_classes, width=head_width, rngs=rngs)

    @classmethod
    def from_pretrained_canvit(
        cls, pretrained_repo: str, *, num_classes: int, head_width: int, seed: int, compute_dtype: ComputeDtype
    ) -> "CanViTForObjectDetection":
        """A pretrained CanViT, from the Hub or a local checkpoint directory, with a new head."""
        pretrained = CanViTForPretraining.from_pretrained(pretrained_repo)
        detector = cls(
            canvit_config=pretrained.canvit.config,
            num_classes=num_classes,
            head_width=head_width,
            rngs=nnx.Rngs(seed),
            compute_dtype=compute_dtype,
        )
        nnx.update(detector.canvit, nnx.state(pretrained.canvit))
        return detector

    @property
    def num_classes(self) -> int:
        return self.head.heatmap.out_features

    def init_state(self, *, batch_size: int, canvas_grid_size: int) -> RecurrentState:
        return self.canvit.init_state(batch_size=batch_size, canvas_grid_size=canvas_grid_size)

    def __call__(
        self, *, glimpse: Array, state: RecurrentState, viewpoint: Viewpoint
    ) -> tuple[CenterNetMaps, RecurrentState]:
        output = self.canvit(glimpse=glimpse, state=state, viewpoint=viewpoint)
        return self.detect(output.state.canvas), output.state

    def detect(self, canvas: Array) -> CenterNetMaps:
        """Decode a canvas into float32 CenterNet maps over its patch grid."""
        return self.head(self.canvit.canvas_patch_grid(canvas).astype(jnp.float32))

    def checkpoint_config(self) -> dict[str, object]:
        return {
            "canvit_config": dataclasses.asdict(self.canvit.config),
            "num_classes": self.num_classes,
            "head_width": self.head.stem.out_features,
        }


def load_detector(repo: str | Path, *, compute_dtype: ComputeDtype) -> CanViTForObjectDetection:
    """A saved detector, from the Hub or a directory, computing in compute_dtype."""
    saved = CanViTForObjectDetection.from_pretrained(repo)
    detector = CanViTForObjectDetection(
        canvit_config=saved.canvit.config,
        num_classes=saved.num_classes,
        head_width=saved.head.stem.out_features,
        rngs=nnx.Rngs(0),
        compute_dtype=compute_dtype,
    )
    nnx.update(detector, nnx.state(saved))
    return detector
