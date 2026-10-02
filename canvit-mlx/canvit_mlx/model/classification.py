from dataclasses import asdict
from typing import Any

import mlx.core as mx
import numpy as np
from canvit_core import CanViTConfig
from canvit_core.checkpoint import hub_file
from canvit_core.readout import fuse_probe
from safetensors.numpy import load_file

from canvit_mlx.hub import HubMixin
from canvit_mlx.model.canvit import CanViT, RecurrentState
from canvit_mlx.model.pretraining import CanViTForPretraining
from canvit_mlx.model.readout import LinearReadout
from canvit_mlx.module import Module
from canvit_mlx.viewpoint import Viewpoint


class CanViTForImageClassification(Module, HubMixin):
    def __init__(self, *, canvit_config: CanViTConfig, n_classes: int) -> None:
        super().__init__()
        assert n_classes > 0, n_classes
        self.canvit = CanViT(canvit_config)
        self.readout = LinearReadout(self.canvit.backbone_dim, n_classes)

    @property
    def n_classes(self) -> int:
        return self.readout.proj.weight.shape[0]

    def checkpoint_config(self) -> dict[str, Any]:
        return dict(canvit_config=asdict(self.canvit.config), n_classes=self.n_classes)

    def init_state(self, *, batch_size: int, canvas_grid_size: int) -> RecurrentState:
        return self.canvit.init_state(batch_size=batch_size, canvas_grid_size=canvas_grid_size)

    def __call__(
        self, *, glimpse: mx.array, state: RecurrentState, viewpoint: Viewpoint
    ) -> tuple[mx.array, RecurrentState]:
        output = self.canvit(glimpse=glimpse, state=state, viewpoint=viewpoint)
        return self.classify(output.state.recurrent_cls), output.state

    def classify(self, recurrent_cls: mx.array) -> mx.array:
        assert recurrent_cls.shape[1] == 1, recurrent_cls.shape
        return self.readout(recurrent_cls[:, 0].astype(mx.float32))

    @classmethod
    def from_pretrained_with_probe(cls, *, pretrained_repo: str, probe_repo: str) -> "CanViTForImageClassification":
        pretrained = CanViTForPretraining.from_pretrained(pretrained_repo)
        probe = load_file(hub_file(probe_repo, "model.safetensors"))
        projection = pretrained.teacher_cls_readout.proj
        standardizer = pretrained.teacher_cls_standardizer
        weight, bias = fuse_probe(
            proj_weight=np.array(projection.weight),
            proj_bias=np.array(projection.bias),
            mean=np.array(standardizer.mean[0]),
            std=np.array(standardizer.std[0]),
            probe_weight=probe["weight"],
            probe_bias=probe["bias"],
        )
        model = cls(canvit_config=pretrained.canvit.config, n_classes=weight.shape[0])
        model.canvit = pretrained.canvit
        model.readout.norm = pretrained.teacher_cls_readout.norm
        model.readout.proj.weight, model.readout.proj.bias = mx.array(weight), mx.array(bias)
        return model
