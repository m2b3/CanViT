"""CanViT with a linear classifier on the recurrent CLS token."""

import torch
from safetensors.torch import load_file
from torch import Tensor, nn

from canvit_pytorch.hub.loading import HubMixin, hub_file
from canvit_pytorch.model.canvit import CanViT, RecurrentState
from canvit_pytorch.model.config import CanViTConfig
from canvit_pytorch.model.pretraining import CanViTForPretraining
from canvit_pytorch.model.readout import LinearReadout
from canvit_pytorch.viewpoint import Viewpoint


def fuse_probe(
    *, proj: nn.Linear, mean: Tensor, std: Tensor, probe_weight: Tensor, probe_bias: Tensor,
) -> tuple[Tensor, Tensor]:
    """Collapse three affine maps into one: project to teacher space, destandardize, apply the teacher's probe.

        logits = W_probe (std ⊙ (W_proj z + b_proj) + mean) + b_probe = W_fused z + b_fused

    Computed in float64; returns (W_fused [n_classes, D], b_fused [n_classes]) in proj's dtype.
    """
    W_proj, b_proj = proj.weight.double(), proj.bias.double()
    mean, std, W_probe, b_probe = mean.double(), std.double(), probe_weight.double(), probe_bias.double()
    teacher_dim = W_proj.shape[0]
    assert mean.shape == std.shape == (teacher_dim,) and W_probe.shape[1] == teacher_dim, (mean.shape, W_probe.shape)
    W_fused = W_probe @ (std.unsqueeze(1) * W_proj)
    b_fused = W_probe @ (std * b_proj + mean) + b_probe
    return W_fused.to(proj.weight.dtype), b_fused.to(proj.weight.dtype)


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
        destandardization; the three affine maps fuse into one (see fuse_probe).
        """
        pretrained = CanViTForPretraining.from_pretrained(pretrained_repo)
        probe = load_file(hub_file(probe_repo, "model.safetensors"))
        standardizer = pretrained.teacher_cls_standardizer
        weight, bias = fuse_probe(
            proj=pretrained.teacher_cls_readout.proj,
            mean=standardizer.mean[0], std=standardizer.std[0],
            probe_weight=probe["weight"], probe_bias=probe["bias"],
        )
        model = cls(canvit_config=pretrained.canvit.config, n_classes=weight.shape[0])
        model.canvit.load_state_dict(pretrained.canvit.state_dict())
        model.readout.norm.load_state_dict(pretrained.teacher_cls_readout.norm.state_dict())
        model.readout.proj.weight.data.copy_(weight)
        model.readout.proj.bias.data.copy_(bias)
        return model
