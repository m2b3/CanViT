"""Training checkpoints: the model, the optimizer and scheduler to resume, and the run's history.

A run directory holds one `step-<N>.pt` per saved step and a `latest.pt`
symlink to the newest; a job resumes from `latest.pt`.
"""

import logging
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypedDict

import torch
from torch import Tensor

from canvit_pytorch import legacy
from canvit_pytorch.model.config import CanViTConfig
from canvit_pytorch.model.pretraining import CanViTForPretraining

log = logging.getLogger(__name__)

CHECKPOINT_FORMAT = "canvit-training-checkpoint-d0354a31-2d71-45e9-b2ad-ded890170034"


class TrainingCheckpoint(TypedDict):
    format: str  # CHECKPOINT_FORMAT
    model: dict[str, Tensor]  # CanViTForPretraining state dict
    canvit_config: dict[str, Any]
    teacher_repo: str
    teacher_dim: int
    teacher_patch_grid: int
    dataset: str
    scene_size_px: int
    glimpse_size_px: int
    step: int
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]
    comet_experiment_key: str
    jobs: list[dict[str, Any]]
    """One entry per job of the run, oldest first: its config and provenance."""


def model_init_kwargs(model: CanViTForPretraining) -> dict[str, Any]:
    return {
        "canvit_config": asdict(model.canvit.config),
        "teacher_dim": model.teacher_dim,
        "teacher_patch_grid": model.teacher_patch_grid,
    }


def model_from_checkpoint(checkpoint: TrainingCheckpoint) -> CanViTForPretraining:
    model = CanViTForPretraining(
        canvit_config=CanViTConfig(**checkpoint["canvit_config"]),
        teacher_dim=checkpoint["teacher_dim"],
        teacher_patch_grid=checkpoint["teacher_patch_grid"],
    )
    model.load_state_dict(checkpoint["model"])
    return model


def save(checkpoint: TrainingCheckpoint, run_dir: Path) -> Path:
    """Write step-<N>.pt atomically and point latest.pt at it."""
    path = run_dir / f"step-{checkpoint['step']}.pt"
    fd, tmp = tempfile.mkstemp(dir=run_dir, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        torch.save(checkpoint, tmp)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)
    link = run_dir / f".latest.pt.{os.getpid()}"
    link.unlink(missing_ok=True)
    link.symlink_to(path.name)
    os.replace(link, run_dir / "latest.pt")
    log.info(f"Saved {path} ({path.stat().st_size / 2**20:.0f} MiB)")
    return path


def latest(run_dir: Path) -> Path | None:
    link = run_dir / "latest.pt"
    if not link.is_symlink():
        return None
    target = link.resolve()
    assert target.exists(), f"{link} points to missing {target}"
    return target


def load(path: Path, *, device: torch.device) -> TrainingCheckpoint:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "state_dict" in checkpoint and "model_config" in checkpoint:
        raise legacy.training_checkpoint_error(path)
    assert checkpoint.get("format") == CHECKPOINT_FORMAT, f"{path} has format {checkpoint.get('format')!r}"
    missing = set(TrainingCheckpoint.__annotations__) - set(checkpoint)
    assert not missing, f"{path} lacks {sorted(missing)}"
    return checkpoint
