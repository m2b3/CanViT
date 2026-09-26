"""Configuration for CanViT pretraining."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

from canvit_pretrain import CanViTForPretrainingConfig
from canvit_pretrain.train.utils import get_sensible_device

# Default HF repo for the teacher model
TEACHER_REPO_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
# Short name used for shard paths and probe lookup (matches precomputed feature directories)
TEACHER_NAME = "dinov3_vitb16"

Objective = Literal["distillation", "rgb_recon"]


@dataclass
class Config:
    # Pretraining objective: DINOv3-feature distillation (flagship) or teacher-free RGB
    # pixel reconstruction (MAE-style control). rgb_recon ignores the teacher entirely.
    objective: Objective = "distillation"
    # Teacher (unused when objective == "rgb_recon")
    teacher_repo_id: str = TEACHER_REPO_ID
    teacher_name: str = TEACHER_NAME
    # Student
    backbone_name: str = "vitb16"
    # Model config (PretrainingConfig via alias)
    # teacher_dim placeholder - overridden by create_model based on actual teacher
    model: CanViTForPretrainingConfig = field(
        default_factory=lambda: CanViTForPretrainingConfig(teacher_dim=768)
    )
    # Glimpse/canvas sizes (runtime, not in model config)
    glimpse_grid_size: int = 8  # tokens per glimpse side
    canvas_patch_grid_size: int = 32  # canvas spatial grid side length in tokens
    # Training
    batch_size: int = 64
    warmup_steps: int = 100_000
    start_lr: float | None = 1e-7  # None = peak_lr / warmup_steps
    peak_lr: float = 4e-4
    cosine_total_steps: int | None = None  # None = constant after warmup; set to enable cosine decay
    weight_decay: float = 1e-4
    min_viewpoint_scale: float = 0.05  # Minimum scale for random viewpoints
    n_full_start_branches: int = 1  # branches starting with FULL viewpoint at t0
    n_random_start_branches: int = 1  # branches starting with RANDOM viewpoint at t0
    chunk_size: int = 2  # BPTT chunk size (glimpses per chunk, gradient flows within)
    continue_prob: float = 0.5  # prob of adding another chunk to trajectory
    enable_scene_patches_loss: bool = True  # Scene (canvas) patch reconstruction loss
    enable_scene_cls_loss: bool = True  # Scene (global) CLS reconstruction loss
    ema_alpha: float = 0.1  # EMA smoothing for metrics
    grad_clip: float = 1.0
    steps_per_job: int = 4_992  # Steps this job does before exiting (for SLURM arrays)
    # Data
    train_dir: Path = Path("/datasets/ILSVRC/Data/CLS-LOC/train")
    val_dir: Path = Path("/datasets/ILSVRC/Data/CLS-LOC/val")
    train_index_dir: Path | None = None  # Required for raw image training
    val_index_dir: Path | None = None  # Required for validation
    # Precomputed features (skips teacher inference on train images)
    # If feature_base_dir is set, shards path is auto-constructed:
    #   {feature_base_dir}/{teacher_name}/{scene_resolution}/shards/
    feature_base_dir: Path | None = None
    feature_image_root: Path | None = None  # Required with feature_base_dir
    tar_dir: Path | None = None  # images read directly from mmap'd tars
    # Run identification and checkpointing
    run_name: str | None = None
    """Run name. Auto-generated from SLURM_ARRAY_JOB_ID or timestamp if None."""
    ckpt_dir: Path = Path("checkpoints")
    """Directory for checkpoint storage. Run checkpoints go in {ckpt_dir}/{run_name}/."""
    seed_ckpt: Path | None = None
    """Seed model weights from external checkpoint (.pt in CheckpointData format).
    Starts fresh (new experiment, step=0). Only used if no checkpoint exists in run_dir."""
    hf_seed_ckpt: str | None = None
    """Seed model weights from HF Hub repo (e.g. '<org>/canvitb16-add-vpe-...'). Downloads
    config.json + model.safetensors, overrides cfg.model with the checkpoint's config.
    Mutually exclusive with seed_ckpt."""
    reset_normalizer: bool = False
    """Re-warmup normalizer stats when loading any checkpoint."""
    normalizer_max_samples: int = 0
    """Max samples from shard for normalizer stats. 0 = use all samples."""
    # Training
    num_workers: int = 16
    scene_resolution: int = 512
    dataset: str = "in21k"
    # Logging
    log_every: int = 20
    val_every: int = 1000
    n_eval_viewpoints: int = 10  # Number of viewpoints in validation (quadtree)
    viz_every_n_vals: int = 5  # Log viz every N validation runs
    curve_every_n_vals: int = 5  # Log curves every N validation runs
    log_spatial_stats: bool = True
    # Compilation and precision
    compile: bool = True
    combo_kernels: bool = False  # torch._inductor.config.combo_kernels (experimental)
    amp: bool = True
    non_blocking_transfer: bool = True  # Ablation: async CPU→GPU transfers
    # Optuna
    n_trials: int = 1
    # Runtime
    device: torch.device = field(default_factory=get_sensible_device)
