"""Shared evaluation configuration.

Dataset paths resolve as env var override > known machine paths > generic
fallback. Real runs validate the resolved paths where the data is opened so
CLI help remains usable on machines without the datasets mounted.
"""

import functools
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from canvit_pytorch.checkpoints import FLAGSHIP_PRETRAIN_REPO
from tqdm import tqdm

from canvit_eval.policies import PolicyName

log = logging.getLogger(__name__)


def progress(iterable, **kw):
    """tqdm, silent when stderr is not a terminal.

    `disable=None` is tqdm's own auto mode and is not the same as
    `disable=False`, which is what a bare tqdm() does.
    """
    return tqdm(iterable, disable=None, **kw)

DEFAULT_PRETRAINED_REPO = FLAGSHIP_PRETRAIN_REPO

# DINOv3 teacher repos (public, third-party — no resolve wrap).
DINOV3_VITB_REPO = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINOV3_VITS_REPO = "facebook/dinov3-vits16-pretrain-lvd1689m"
TEACHER_REPO = DINOV3_VITB_REPO


def _resolve_path(env_var: str, known_paths: list[str], description: str) -> Path:
    """Resolve a dataset path without touching the filesystem on fallback."""
    v = os.environ.get(env_var)
    if v is not None:
        log.info("%s from env: %s", description, v)
        return Path(v)
    for p in known_paths:
        if Path(p).is_dir():
            log.info("%s autodetected: %s", description, p)
            return Path(p)
    fallback = Path(known_paths[0])
    log.info("%s defaulting to %s; validate before use", description, fallback)
    return fallback


def require_existing_dir(path: Path, *, description: str, env_var: str | None = None) -> None:
    if path.is_dir():
        return
    hint = f" Set {env_var} or pass the corresponding CLI path." if env_var else ""
    raise RuntimeError(f"{description} not found at {path}.{hint}")


@functools.cache
def ade20k_root() -> Path:
    return _resolve_path(
        env_var="ADE20K_ROOT",
        known_paths=["/datasets/ADE20k/ADEChallengeData2016"],
        description="ADE20K root",
    )


@functools.cache
def imagenet_val_dir() -> Path:
    return _resolve_path(
        env_var="IMAGENET_VAL",
        known_paths=[
            "/datasets/ILSVRC/Data/CLS-LOC/val",
            "/datashare/imagenet/ILSVRC2012/val",
        ],
        description="ImageNet val dir",
    )


@dataclass(frozen=True)
class EpisodeConfig:
    """How to run CanViT episodes. Shared by all CanViT-based tasks."""

    model_repo: str = DEFAULT_PRETRAINED_REPO
    policy: PolicyName = "coarse_to_fine"
    n_timesteps: int = 21
    canvas_grid: int | None = None  # None → scene_size // patch_size
    glimpse_px: int = 128
    min_scale: float = 0.05
    max_scale: float = 1.0
