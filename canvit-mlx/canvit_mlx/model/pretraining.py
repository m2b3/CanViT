import math
from dataclasses import asdict
from typing import Any

import mlx.core as mx
from canvit_core import CanViTConfig

from canvit_mlx.hub import HubMixin
from canvit_mlx.model.canvit import CanViT, CanViTOutput, RecurrentState
from canvit_mlx.model.readout import LinearReadout
from canvit_mlx.model.standardizer import PositionAwareStandardizer
from canvit_mlx.module import Module
from canvit_mlx.viewpoint import Viewpoint


class CanViTForPretraining(Module, HubMixin):
    def __init__(self, *, canvit_config: CanViTConfig, teacher_dim: int, teacher_patch_grid: int) -> None:
        super().__init__()
        assert min(teacher_dim, teacher_patch_grid) > 0, (teacher_dim, teacher_patch_grid)
        self.canvit = CanViT(canvit_config)
        self.teacher_dim = teacher_dim
        self.teacher_patch_grid = teacher_patch_grid
        self.teacher_patch_readout = LinearReadout(self.canvit.canvas_dim, teacher_dim)
        self.teacher_cls_readout = LinearReadout(self.canvit.backbone_dim, teacher_dim)
        for readout in (self.teacher_patch_readout, self.teacher_cls_readout):
            readout.norm.weight = mx.ones_like(readout.norm.weight) / math.sqrt(readout.norm.weight.size)
        self.teacher_patch_standardizer = PositionAwareStandardizer(teacher_patch_grid**2, teacher_dim)
        self.teacher_cls_standardizer = PositionAwareStandardizer(1, teacher_dim)

    def checkpoint_config(self) -> dict[str, Any]:
        return dict(
            canvit_config=asdict(self.canvit.config),
            teacher_dim=self.teacher_dim,
            teacher_patch_grid=self.teacher_patch_grid,
        )

    def init_state(self, *, batch_size: int, canvas_grid_size: int) -> RecurrentState:
        return self.canvit.init_state(batch_size=batch_size, canvas_grid_size=canvas_grid_size)

    def __call__(self, *, glimpse: mx.array, state: RecurrentState, viewpoint: Viewpoint) -> CanViTOutput:
        return self.canvit(glimpse=glimpse, state=state, viewpoint=viewpoint)

    def predict_teacher_patches(self, canvas: mx.array) -> mx.array:
        return self.teacher_patch_readout(self.canvit.canvas_patches(canvas))

    def predict_teacher_cls(self, recurrent_cls: mx.array) -> mx.array:
        assert recurrent_cls.shape[1] == 1, recurrent_cls.shape
        return self.teacher_cls_readout(recurrent_cls[:, 0])
