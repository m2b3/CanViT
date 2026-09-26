"""Validation rollouts on held-out ImageNet-1k images, glimpse by glimpse.

After each glimpse: cosine similarity of the predictions to the teacher's
features, and ImageNet-1k top-1 accuracy of the teacher's linear classifier
applied to the predicted CLS token, next to the same classifier on the
teacher's own CLS token.
"""

from contextlib import AbstractContextManager
from dataclasses import dataclass

import torch
from safetensors.torch import load_file
from torch import Tensor, nn

from canvit_pytorch.episode import run_episode
from canvit_pytorch.hub.loading import hub_file
from canvit_pytorch.hub.repos import DINOV3_VITB16_IN1K_PROBE
from canvit_pytorch.model.pretraining import CanViTForPretraining
from canvit_pytorch.policies import PolicyName, make_policy
from canvit_pytorch.pretrain.loss import DistillationTargets, teacher_similarities
from canvit_pytorch.teacher import DINOv3Teacher
from canvit_pytorch.viewpoint import Viewpoint

TEACHER_CLASSIFIER_SIZE_PX = 512
"""The resolution DINOV3_VITB16_IN1K_PROBE was trained at."""


def load_teacher_classifier(device: torch.device) -> nn.Linear:
    """DINOv3 ViT-B/16's linear ImageNet-1k classifier on its CLS token."""
    weights = load_file(hub_file(DINOV3_VITB16_IN1K_PROBE, "model.safetensors"))
    classifier = nn.Linear(weights["weight"].shape[1], weights["weight"].shape[0])
    classifier.load_state_dict(weights)
    return classifier.to(device).eval().requires_grad_(False)


def _top1(logits: Tensor, labels: Tensor) -> float:
    return (logits.argmax(dim=-1) == labels).float().mean().item()


@dataclass(frozen=True)
class FigureInputs:
    """The first scene of the batch, for monitor.figures.rollout_figure."""

    image: Tensor
    viewpoints: list[Viewpoint]
    teacher_patches: Tensor
    predicted_patches: list[Tensor]
    canvas_patches: list[Tensor]


@dataclass(frozen=True)
class ValidationRollout:
    per_glimpse: list[dict[str, float]]
    teacher_in1k_top1: float
    figure_inputs: FigureInputs


@torch.inference_mode()
def validate(
    *,
    model: CanViTForPretraining,
    teacher: DINOv3Teacher,
    teacher_classifier: nn.Linear,
    images: Tensor,
    labels: Tensor,
    policy_name: PolicyName,
    num_glimpses: int,
    glimpse_size_px: int,
    canvas_grid_size: int,
    autocast: AbstractContextManager,
) -> ValidationRollout:
    was_training = model.training
    model.eval()
    try:
        with autocast:
            features = teacher(images)
        targets = DistillationTargets.standardize(model, patches=features.patches.float(), cls=features.cls.float())
        policy = make_policy(
            policy_name, batch_size=images.shape[0], device=images.device, num_glimpses=num_glimpses,
            canvas_grid_size=canvas_grid_size,
        )
        with autocast:
            steps = run_episode(
                canvit=model.canvit, images=images, policy=policy, num_glimpses=num_glimpses,
                glimpse_size_px=glimpse_size_px,
                initial_state=model.init_state(batch_size=images.shape[0], canvas_grid_size=canvas_grid_size),
            )
        per_glimpse = []
        for step in steps:
            metrics = {name: value.item() for name, value in teacher_similarities(model, step.state, targets).items()}
            predicted_cls = model.predict_teacher_cls(step.state.recurrent_cls)
            raw_cls = model.teacher_cls_standardizer.destandardize(predicted_cls.unsqueeze(1)).squeeze(1)
            per_glimpse.append(metrics | {"in1k_top1": _top1(teacher_classifier(raw_cls), labels)})
        figure_inputs = FigureInputs(
            image=images[0].cpu(),
            viewpoints=[Viewpoint(centers=s.viewpoint.centers[:1].cpu(), scales=s.viewpoint.scales[:1].cpu()) for s in steps],
            teacher_patches=targets.patches[0].cpu(),
            predicted_patches=[model.predict_teacher_patches(s.state.canvas[:1])[0].cpu() for s in steps],
            canvas_patches=[model.canvit.canvas_patches(s.state.canvas[:1])[0].cpu() for s in steps],
        )
        return ValidationRollout(
            per_glimpse=per_glimpse,
            teacher_in1k_top1=_top1(teacher_classifier(targets.raw_cls), labels),
            figure_inputs=figure_inputs,
        )
    finally:
        model.train(was_training)
