"""CanViT with a linear classifier on the recurrent CLS token."""

import torch
from canvit_core.checkpoint import hub_file
from canvit_core.config import CanViTConfig
from canvit_core.readout import fuse_probe
from safetensors.numpy import load_file
from torch import Tensor, nn

from canvit_pytorch.hub.loading import HubMixin
from canvit_pytorch.model.canvit import CanViT, RecurrentState
from canvit_pytorch.model.pretraining import CanViTForPretraining
from canvit_pytorch.model.readout import LinearReadout
from canvit_pytorch.viewpoint import Viewpoint


class CanViTForImageClassification(nn.Module, HubMixin):
    def __init__(self, *, canvit_config: CanViTConfig, n_classes: int) -> None:
        super().__init__()
        self.canvit = CanViT(canvit_config)
        self.readout = LinearReadout(self.canvit.backbone_dim, n_classes)

    @property
    def n_classes(self) -> int:
        return self.readout.proj.out_features

    def init_state(self, *, batch_size: int, canvas_grid_size: int) -> RecurrentState:
        return self.canvit.init_state(batch_size=batch_size, canvas_grid_size=canvas_grid_size)

    def forward(self, *, glimpse: Tensor, state: RecurrentState, viewpoint: Viewpoint) -> tuple[Tensor, RecurrentState]:
        """(logits [B, n_classes], next state)."""
        out = self.canvit(glimpse=glimpse, state=state, viewpoint=viewpoint)
        return self.classify(out.state.recurrent_cls), out.state

    def classify(self, recurrent_cls: Tensor) -> Tensor:
        """[B, 1, backbone_dim] -> [B, n_classes] float32 logits, also under autocast."""
        with torch.autocast(device_type=recurrent_cls.device.type, enabled=False):
            return self.readout(recurrent_cls[:, 0].float())

    @classmethod
    def from_pretrained_with_probe(cls, *, pretrained_repo: str, probe_repo: str) -> "CanViTForImageClassification":
        """A pretrained CanViT whose CLS readout is fused with a linear probe trained on the teacher's CLS token.

        Pretraining taught the recurrent CLS token to predict the teacher's CLS
        token, so a probe fitted on the teacher applies to CanViT after
        destandardization; the three affine maps fuse into one (canvit_core.readout.fuse_probe).
        """
        pretrained = CanViTForPretraining.from_pretrained(pretrained_repo)
        probe = load_file(hub_file(probe_repo, "model.safetensors"))
        projection, standardizer = pretrained.teacher_cls_readout.proj, pretrained.teacher_cls_standardizer
        weight, bias = fuse_probe(
            proj_weight=projection.weight.detach().numpy(), proj_bias=projection.bias.detach().numpy(),
            mean=standardizer.mean[0].numpy(), std=standardizer.std[0].numpy(),
            probe_weight=probe["weight"], probe_bias=probe["bias"],
        )
        model = cls(canvit_config=pretrained.canvit.config, n_classes=weight.shape[0])
        model.canvit.load_state_dict(pretrained.canvit.state_dict())
        model.readout.norm.load_state_dict(pretrained.teacher_cls_readout.norm.state_dict())
        model.readout.proj.weight.data.copy_(torch.from_numpy(weight))
        model.readout.proj.bias.data.copy_(torch.from_numpy(bias))
        return model
