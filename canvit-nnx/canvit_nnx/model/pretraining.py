"""CanViT readouts for dense teacher-feature distillation."""

import dataclasses
import math

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from canvit_core.config import CanViTConfig

from canvit_nnx.hub import HubMixin
from canvit_nnx.model.canvit import CanViT, CanViTOutput, RecurrentState
from canvit_nnx.model.readout import LinearReadout
from canvit_nnx.model.standardizer import PositionAwareStandardizer
from canvit_nnx.viewpoint import Viewpoint

Array = jax.Array


class CanViTForPretraining(nnx.Module, HubMixin):
    def __init__(
        self,
        *,
        canvit_config: CanViTConfig,
        teacher_dim: int,
        teacher_patch_grid: int,
        rngs: nnx.Rngs,
    ) -> None:
        assert teacher_dim > 0 and teacher_patch_grid > 0, (teacher_dim, teacher_patch_grid)
        self.canvit = CanViT(canvit_config, rngs=rngs)
        self.teacher_dim = teacher_dim
        self.teacher_patch_grid = teacher_patch_grid
        self.teacher_patch_readout = LinearReadout(
            in_dim=self.canvit.canvas_dim,
            out_dim=teacher_dim,
            rngs=rngs,
        )
        self.teacher_cls_readout = LinearReadout(
            in_dim=self.canvit.backbone_dim,
            out_dim=teacher_dim,
            rngs=rngs,
        )
        for readout in (self.teacher_patch_readout, self.teacher_cls_readout):
            assert readout.norm.scale is not None
            readout.norm.scale[...] = jnp.full(
                (readout.norm.num_features,),
                1.0 / math.sqrt(readout.norm.num_features),
            )
        self.teacher_patch_standardizer = PositionAwareStandardizer(
            n_positions=teacher_patch_grid**2,
            dim=teacher_dim,
        )
        self.teacher_cls_standardizer = PositionAwareStandardizer(n_positions=1, dim=teacher_dim)

    def init_state(self, *, batch_size: int, canvas_grid_size: int) -> RecurrentState:
        return self.canvit.init_state(batch_size=batch_size, canvas_grid_size=canvas_grid_size)

    def __call__(self, *, glimpse: Array, state: RecurrentState, viewpoint: Viewpoint) -> CanViTOutput:
        return self.canvit(glimpse=glimpse, state=state, viewpoint=viewpoint)

    def predict_teacher_patches(self, canvas: Array) -> Array:
        return self.teacher_patch_readout(self.canvit.canvas_patches(canvas))

    def predict_teacher_cls(self, recurrent_cls: Array) -> Array:
        assert recurrent_cls.ndim == 3 and recurrent_cls.shape[1] == 1, recurrent_cls.shape
        return self.teacher_cls_readout(recurrent_cls[:, 0, :])

    def checkpoint_config(self) -> dict[str, object]:
        return {
            "canvit_config": dataclasses.asdict(self.canvit.config),
            "teacher_dim": self.teacher_dim,
            "teacher_patch_grid": self.teacher_patch_grid,
        }
