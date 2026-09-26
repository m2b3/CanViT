"""CanViT with the readouts that pretraining fits to a frozen DINOv3 teacher.

Passive-to-active dense latent distillation: after every glimpse, the canvas is
decoded into the teacher's patch features of the whole scene and the recurrent
CLS token into the teacher's CLS token, both in per-position standardized space.
"""

import math

from torch import Tensor, nn

from canvit_pytorch.hub.loading import HubMixin
from canvit_pytorch.model.canvit import CanViT, CanViTOutput, RecurrentState
from canvit_pytorch.model.config import CanViTConfig
from canvit_pytorch.model.readout import LinearReadout
from canvit_pytorch.model.standardizer import PositionAwareStandardizer
from canvit_pytorch.viewpoint import Viewpoint


class CanViTForPretraining(nn.Module, HubMixin):
    def __init__(self, *, canvit_config: CanViTConfig, teacher_dim: int, teacher_patch_grid: int) -> None:
        """teacher_patch_grid: side of the teacher's patch grid, which the canvas must match during pretraining."""
        super().__init__()
        self.canvit = CanViT(canvit_config)
        self.teacher_dim = teacher_dim
        self.teacher_patch_grid = teacher_patch_grid
        self.teacher_patch_readout = LinearReadout(self.canvit.canvas_dim, teacher_dim)
        self.teacher_cls_readout = LinearReadout(self.canvit.backbone_dim, teacher_dim)
        for readout in (self.teacher_patch_readout, self.teacher_cls_readout):
            readout.norm.weight.data.fill_(1.0 / math.sqrt(readout.norm.normalized_shape[0]))
        self.teacher_patch_standardizer = PositionAwareStandardizer(teacher_patch_grid**2, teacher_dim)
        self.teacher_cls_standardizer = PositionAwareStandardizer(1, teacher_dim)

    def init_state(self, *, batch_size: int, canvas_grid_size: int) -> RecurrentState:
        return self.canvit.init_state(batch_size=batch_size, canvas_grid_size=canvas_grid_size)

    def forward(self, *, glimpse: Tensor, state: RecurrentState, viewpoint: Viewpoint) -> CanViTOutput:
        return self.canvit(glimpse=glimpse, state=state, viewpoint=viewpoint)

    def predict_teacher_patches(self, canvas: Tensor) -> Tensor:
        """[B, G*G, teacher_dim], standardized."""
        return self.teacher_patch_readout(self.canvit.canvas_patches(canvas)).contiguous()

    def predict_teacher_cls(self, recurrent_cls: Tensor) -> Tensor:
        """[B, 1, backbone_dim] -> [B, teacher_dim], standardized."""
        assert recurrent_cls.shape[1] == 1, recurrent_cls.shape
        return self.teacher_cls_readout(recurrent_cls[:, 0]).contiguous()
