"""One pretraining step (paper, Section 5.2): rollouts over a batch of scenes, trained with truncated BPTT.

Each rollout starts from a fresh canvas and takes its viewpoints from a viewing
policy (F-IID and R-IID by default). Its length is random: whole chunks of K
glimpses, stopping after each chunk with probability p_stop. Gradients flow
within a chunk; the state is detached between chunks, so memory stays that of
K glimpses. Every glimpse's loss counts, averaged over glimpses and rollouts.
"""

import random
from contextlib import AbstractContextManager
from dataclasses import dataclass

import torch
from torch import Tensor

from canvit_pytorch.model.pretraining import CanViTForPretraining
from canvit_pytorch.policies import PolicyName, make_policy
from canvit_pytorch.pretrain.loss import DistillationTargets, distillation_losses, teacher_similarities
from canvit_pytorch.viewpoint import sample_at_viewpoint


def sample_rollout_length(*, tbptt_chunk_glimpses: int, stop_probability: float) -> int:
    num_glimpses = tbptt_chunk_glimpses
    while random.random() < 1 - stop_probability:
        num_glimpses += tbptt_chunk_glimpses
    return num_glimpses


@dataclass(frozen=True)
class StepMetrics:
    loss: Tensor  # mean over rollouts of each rollout's mean loss per glimpse
    num_glimpses: int
    by_policy: dict[PolicyName, dict[str, Tensor]]
    """Per viewing policy, averaged over its rollouts: the loss and each loss term per glimpse, and the
    teacher similarities after the last glimpse."""


def training_step(
    *,
    model: CanViTForPretraining,
    images: Tensor,
    targets: DistillationTargets,
    rollout_policies: tuple[PolicyName, ...],
    tbptt_chunk_glimpses: int,
    stop_probability: float,
    glimpse_size_px: int,
    canvas_grid_size: int,
    enable_teacher_patch_loss: bool,
    enable_teacher_cls_loss: bool,
    autocast: AbstractContextManager,
) -> StepMetrics:
    """Run every rollout on ImageNet-normalized images [B, 3, H, W], accumulating gradients; the caller steps the optimizer."""
    num_glimpses = sample_rollout_length(tbptt_chunk_glimpses=tbptt_chunk_glimpses, stop_probability=stop_probability)
    batch_size, device = images.shape[0], images.device
    rollouts: list[tuple[PolicyName, dict[str, Tensor]]] = []
    for policy_name in rollout_policies:
        policy = make_policy(
            policy_name, batch_size=batch_size, device=device, num_glimpses=num_glimpses, canvas_grid_size=canvas_grid_size,
        )
        state = model.init_state(batch_size=batch_size, canvas_grid_size=canvas_grid_size)
        chunk_loss = torch.zeros((), device=device)
        loss_sums: dict[str, Tensor] = {}
        for t in range(num_glimpses):
            with autocast:
                viewpoint = policy.step(t, state)
                glimpse = sample_at_viewpoint(spatial=images, viewpoint=viewpoint, glimpse_size_px=glimpse_size_px)
                state = model(glimpse=glimpse, state=state, viewpoint=viewpoint).state
                losses = distillation_losses(
                    model, state, targets,
                    enable_teacher_patch_loss=enable_teacher_patch_loss, enable_teacher_cls_loss=enable_teacher_cls_loss,
                )
            loss = torch.stack(list(losses.values())).sum().float()
            chunk_loss = chunk_loss + loss
            for name, value in {"loss": loss, **losses}.items():
                loss_sums[name] = loss_sums.get(name, 0.0) + value.detach().float()
            if (t + 1) % tbptt_chunk_glimpses == 0:
                (chunk_loss / num_glimpses / len(rollout_policies)).backward()
                chunk_loss = torch.zeros((), device=device)
                state = state.detach()
        metrics = {name: total / num_glimpses for name, total in loss_sums.items()}
        rollouts.append((policy_name, metrics | teacher_similarities(model, state, targets)))

    by_policy: dict[PolicyName, dict[str, Tensor]] = {
        name: {key: torch.stack([m[key] for n, m in rollouts if n == name]).mean() for key in rollouts[0][1]}
        for name in dict.fromkeys(rollout_policies)
    }
    loss = torch.stack([m["loss"] for _, m in rollouts]).mean()
    return StepMetrics(loss=loss, num_glimpses=num_glimpses, by_policy=by_policy)
