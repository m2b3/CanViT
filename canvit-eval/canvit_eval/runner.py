"""Per-batch episode generator shared across tasks."""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from canvit_pytorch.model.pretraining.hub import CanViTForPretrainingHFHub
from torch import Tensor
from torch.utils.data import DataLoader

from canvit_eval.config import EpisodeConfig, progress
from canvit_eval.episode import CanViTModel, EpisodeStep, run_episode
from canvit_eval.policies import make_policy

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchResult:
    steps: list[EpisodeStep]
    batch: tuple[Tensor, ...]


def load_model(model_repo: str, device: torch.device) -> CanViTForPretrainingHFHub:
    log.info("Loading model: %s", model_repo)
    model = CanViTForPretrainingHFHub.from_pretrained(model_repo).to(device).eval()
    log.info("  canvas_dim=%d, local_dim=%d", model.canvas_dim, model.local_dim)
    return model


def eval_batches(
    *,
    model: CanViTModel,
    loader: DataLoader,
    episode_cfg: EpisodeConfig,
    canvas_grid: int,
    device: torch.device,
    amp: bool = True,
    policy_kwargs: dict[str, Any] | None = None,
) -> Iterator[BatchResult]:
    T = episode_cfg.n_timesteps
    amp_dtype = torch.bfloat16 if amp else torch.float32
    kw = policy_kwargs or {}

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
        for batch in progress(loader, desc="Evaluating"):
            images = batch[0].to(device, non_blocking=True)
            B = images.shape[0]

            policy = make_policy(
                episode_cfg.policy, B, device, T, canvas_grid=canvas_grid,
                min_scale=episode_cfg.min_scale, max_scale=episode_cfg.max_scale,
                **kw,
            )
            steps = run_episode(
                model=model, images=images, policy=policy,
                n_timesteps=T, canvas_grid=canvas_grid, glimpse_px=episode_cfg.glimpse_px,
            )
            yield BatchResult(steps=steps, batch=batch)
