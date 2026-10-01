"""One glimpse of CanViT, and the readout of its canvas by a probe, as modules on plain tensors for export to ONNX.

The browser runs the step graph once per glimpse, then the probe graph on the new canvas. The state goes in and
comes out, so the page keeps it between calls; the glimpse is cropped inside the step graph, so the scene is
uploaded once. CanViT and the probe are exported and published separately.
"""

from typing import NamedTuple

import torch
from torch import Tensor, nn

from canvit_pytorch.model.canvit import CanViT, RecurrentState
from canvit_pytorch.model.segmentation import CanViTForSemanticSegmentation
from canvit_pytorch.viewpoint import Viewpoint, sample_at_viewpoint


class StepInputs(NamedTuple):
    scene: Tensor  # [1, 3, S, S], ImageNet-normalized
    canvas: Tensor  # [1, num_canvas_registers + G*G, canvas_dim]
    recurrent_cls: Tensor  # [1, 1, backbone_dim]
    centers: Tensor  # [1, 2]: the viewpoint's (row, col)
    scales: Tensor  # [1]


class StepOutputs(NamedTuple):
    next_canvas: Tensor
    next_recurrent_cls: Tensor
    glimpse: Tensor  # [1, 3, g, g]: the model's input, ImageNet-normalized


class ReadoutInputs(NamedTuple):
    canvas: Tensor  # [1, num_canvas_registers + G*G, canvas_dim]


class ReadoutOutputs(NamedTuple):
    logits: Tensor  # [1, num_classes, G, G]
    entropy: Tensor  # [1, G, G]: predictive entropy of each cell, in nats


def predictive_entropy_via_logsumexp(logits: Tensor) -> Tensor:
    """policies.entropy.predictive_entropy with log_softmax written as logits - logsumexp: ONNX Runtime Web
    1.30.0's WebGPU execution provider has no LogSoftmax kernel, and running one on the CPU would copy the
    logits off the GPU in the middle of every step."""
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(log_probs.exp() * log_probs).sum(dim=1)


class GlimpseStep(nn.Module):
    def __init__(self, canvit: CanViT, *, glimpse_size_px: int) -> None:
        super().__init__()
        self.canvit = canvit
        self.glimpse_size_px = glimpse_size_px

    def forward(self, scene: Tensor, canvas: Tensor, recurrent_cls: Tensor, centers: Tensor, scales: Tensor) -> StepOutputs:
        viewpoint = Viewpoint(centers=centers, scales=scales)
        glimpse = sample_at_viewpoint(spatial=scene, viewpoint=viewpoint, glimpse_size_px=self.glimpse_size_px)
        state = self.canvit(glimpse=glimpse, state=RecurrentState(canvas=canvas, recurrent_cls=recurrent_cls),
                            viewpoint=viewpoint).state
        return StepOutputs(state.canvas, state.recurrent_cls, glimpse)


class ProbeReadout(nn.Module):
    """The probe on a canvas, through CanViTForSemanticSegmentation.logits, the code path PyTorch users run; only the
    probe's weights reach the exported graph."""

    def __init__(self, model: CanViTForSemanticSegmentation) -> None:
        super().__init__()
        assert isinstance(model.probe.ln, nn.LayerNorm), "the probe must decode the layer-normalized canvas"
        self.model = model

    def forward(self, canvas: Tensor) -> ReadoutOutputs:
        logits = self.model.logits(canvas)
        return ReadoutOutputs(logits, predictive_entropy_via_logsumexp(logits))


def initial_inputs(canvit: CanViT, scene: Tensor, *, canvas_grid_size: int) -> StepInputs:
    """The first glimpse's inputs, at the full scene."""
    state = canvit.init_state(batch_size=1, canvas_grid_size=canvas_grid_size)
    viewpoint = Viewpoint.full_scene(batch_size=1, device=scene.device)
    return StepInputs(
        scene, state.canvas.detach().clone(), state.recurrent_cls.detach().clone(), viewpoint.centers, viewpoint.scales,
    )
