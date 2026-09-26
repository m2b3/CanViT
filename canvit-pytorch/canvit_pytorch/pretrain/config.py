"""Pretraining settings. The defaults are CanViT-B's (paper, Appendix, CanViT-B pretraining hyperparameters)."""

from dataclasses import dataclass
from pathlib import Path

from canvit_pytorch.hub.repos import RELEASED_CANVAS_GRID_SIZE, RELEASED_GLIMPSE_SIZE_PX, PretrainingDataset
from canvit_pytorch.model.config import CanViTConfig
from canvit_pytorch.policies import POLICIES, PolicyName


@dataclass(frozen=True)
class PretrainingConfig:
    shards_dir: Path
    """Teacher features of the training scenes, written by `python -m canvit_pytorch.pretrain.features.export`."""
    images_dir: Path
    """The directory the shards' image paths are relative to."""
    validation_dir: Path
    """ImageNet-1k validation images, one subdirectory per class; they monitor the run."""
    dataset: PretrainingDataset
    """Recorded in checkpoints; released checkpoints are named after it."""
    checkpoints_dir: Path
    run_name: str
    """Checkpoints go to checkpoints_dir/run_name. A job that finds one there resumes from it."""

    model: CanViTConfig = CanViTConfig()
    glimpse_size_px: int = RELEASED_GLIMPSE_SIZE_PX
    canvas_grid_size: int = RELEASED_CANVAS_GRID_SIZE
    """Side of the canvas grid in tokens. It equals the teacher's patch grid, so scenes are canvas_grid_size × 16 px."""

    rollout_policies: tuple[PolicyName, ...] = ("full_then_random", "random")
    """One rollout per policy and scene, each from a fresh canvas; their losses are averaged."""
    tbptt_chunk_glimpses: int = 2
    """K: gradients flow through chunks of K consecutive glimpses (truncated BPTT)."""
    stop_probability: float = 0.5
    """After each chunk, the rollout stops with this probability: rollouts average K / stop_probability glimpses."""
    enable_teacher_patch_loss: bool = True
    """Dense supervision: predict the teacher's patch features of the whole scene from the canvas."""
    enable_teacher_cls_loss: bool = True
    """Predict the teacher's CLS token from the recurrent CLS token."""

    batch_size: int = 64
    start_lr: float = 1e-7
    peak_lr: float = 4e-4
    warmup_steps: int = 100_000
    """Linear warmup from start_lr to peak_lr, then constant."""
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    total_steps: int = 2_000_000
    steps_per_job: int = 4_992
    """Each job trains this many steps, saves a checkpoint and exits; a SLURM array chains the jobs."""

    device: str = "cuda"
    compile: bool = True
    amp: bool = True
    """bfloat16 autocast."""
    num_workers: int = 16

    log_every: int = 20
    validate_every: int = 1_000
    figure_every: int = 5_000
    validation_policy: PolicyName = "coarse_to_fine"
    validation_glimpses: int = 10
    metric_ema_alpha: float = 0.1
    """Weight of the newest step in the logged moving averages of training metrics."""

    def __post_init__(self) -> None:
        assert self.rollout_policies, "at least one rollout per scene"
        for name in (*self.rollout_policies, self.validation_policy):
            assert not POLICIES[name].needs_segmentation_probe, f"{POLICIES[name].paper_name} needs a segmentation probe"
        assert self.tbptt_chunk_glimpses >= 1, self.tbptt_chunk_glimpses
        assert 0 < self.stop_probability <= 1, self.stop_probability
        assert self.enable_teacher_patch_loss or self.enable_teacher_cls_loss, "enable at least one loss"
        assert 1 <= self.warmup_steps <= self.total_steps, (self.warmup_steps, self.total_steps)
        assert self.figure_every % self.validate_every == 0, "figures come from validation rollouts"
