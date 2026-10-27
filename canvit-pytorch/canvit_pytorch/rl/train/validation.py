"""The deployed policy's segmentation cross-entropy after each glimpse, by which a training run selects its scorer."""

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from canvit_pytorch.rl.policy import LearnedPolicy
from canvit_pytorch.rl.train.reward import segmentation_cross_entropy


@torch.no_grad()
def cross_entropy_by_glimpse(
    policy: LearnedPolicy, loader: DataLoader[tuple[Tensor, Tensor]], *, num_glimpses: int, device: torch.device,
    amp: bool,
) -> list[float]:
    """The mean over the loader's scenes of the probe's cross-entropy after each glimpse, at the labels' resolution.
    CanViT runs under bfloat16 autocast with amp."""
    totals = torch.zeros(num_glimpses, device=device)
    num_scenes = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            steps = policy.rollout(images, num_glimpses=num_glimpses)
        for step in steps:
            logits = policy.model.logits(step.state.canvas)
            totals[step.t] += segmentation_cross_entropy(logits, labels, resolution_px=labels.shape[-1]).sum()
        num_scenes += images.shape[0]
    return (totals / num_scenes).tolist()
