"""The paper's ADE20K probe-training protocol (Appendix D.3) and a training run's bookkeeping."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from canvit_pytorch.hub.repos import RELEASED_PROBE_STEPS, RELEASED_SCENE_SIZE_PX


@dataclass(frozen=True)
class ProbeTrainingConfig:
    """Defaults are the paper's protocol, which takes the schedule and augmentation of DINOv3's linear
    ADE20K evaluation (dinov3/eval/segmentation/configs/config-ade20k-linear-training.yaml)."""

    output_dir: Path
    """Each run writes its selected probe and its record to a new subdirectory of output_dir."""
    device: str = "cuda"
    seed: int = 0
    """Seeds the probe's initialization, data order, augmentation, viewpoints and dropout."""

    scene_size_px: int = RELEASED_SCENE_SIZE_PX
    """Side of the square training crops and of the resized validation images."""
    scale_jitter_range: tuple[float, float] = (0.5, 2.0)
    """A training image's short side is resized to scene_size_px times a factor drawn from this range, then cropped."""
    horizontal_flip_probability: float = 0.5

    num_steps: int = RELEASED_PROBE_STEPS
    batch_size: int = 16
    peak_lr: float = 3e-4
    weight_decay: float = 1e-3
    warmup_steps: int = 1500
    warmup_start_lr_fraction: float = 1e-6
    """The learning rate at the first step, as a fraction of peak_lr; after warmup it decays to zero along a cosine."""
    dropout: float = 0.1

    feature_dtype: Literal["bfloat16", "float32"] = "bfloat16"
    """Autocast dtype of the frozen model that computes the features; the probe computes in float32."""
    validation_batch_size: int = 32
    num_workers: int = 4
    validate_every: int = 500
    """Validation also runs after the last step; the run keeps the probe of highest validation mIoU on the last map."""
    log_every: int = 20
    figure_every: int = 500
    figure_samples: int = 4
    comet_project: str = "canvit-ade20k-probes"
