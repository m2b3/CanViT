"""ImageNet-1k classification via CanViTForImageClassification.

Supports both pretrained (frozen probe, fused at construction) and finetuned models.
"""

import logging
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader

from canvit_pytorch import CanViTForImageClassification, resolve_canvit_repo
from canvit_specialize.training.utils import collect_metadata

from canvit_eval.config import DEFAULT_PRETRAINED_REPO, EpisodeConfig, imagenet_val_dir, require_existing_dir
from canvit_eval.datasets.imagenet import make_in1k_dataset
from canvit_eval.runner import eval_batches
from canvit_eval.tasks.base import TaskConfig

log = logging.getLogger(__name__)
TOP_K = 5

DEFAULT_FINETUNED_REPO = resolve_canvit_repo("canvitb16-add-vpe-finetune-g128px-s512px-in1k-2026-04-06")
DEFAULT_PROBE_REPO = resolve_canvit_repo("dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe")

# Flagship CanViT-B was pretrained at canvas_grid=32 only — its CLS standardizer
# is initialized for c=32 and undefined for other grids. Fusion (proj → destandardize
# → probe) therefore always uses the c=32 stats, regardless of runtime canvas_grid.
# This lets us sweep inference canvas_grid without retraining the probe.
FUSION_CANVAS_GRID = 32


@dataclass(kw_only=True)
class Config(TaskConfig):
    output: Path = Path("results/in1k_clf.pt")
    batch_size: int = 64
    mode: Literal["finetuned", "frozen"] = "finetuned"
    scene_size: int = 512
    """Image-space resolution (input images are resized to scene_size × scene_size).
    Tracked alongside canvas_grid so downstream exporters can group runs by (scene, grid)."""
    episode: EpisodeConfig = field(default_factory=lambda: EpisodeConfig(canvas_grid=32))
    probe_repo: str = DEFAULT_PROBE_REPO
    val_dir: Path = field(default_factory=imagenet_val_dir)

    def __post_init__(self) -> None:
        # Auto-swap the pretrained repo for the finetuned one in finetuned mode,
        # but only when the caller hasn't already overridden it.
        if self.mode == "finetuned" and self.episode.model_repo == DEFAULT_PRETRAINED_REPO:
            self.episode = replace(self.episode, model_repo=DEFAULT_FINETUNED_REPO)

    def run(self) -> Path:
        return evaluate(self)


def _load_classifier(cfg: Config, device: torch.device) -> CanViTForImageClassification:
    if cfg.mode == "finetuned":
        clf = CanViTForImageClassification.from_pretrained(cfg.episode.model_repo)
        log.info("Loaded finetuned classifier from %s (%d classes)", cfg.episode.model_repo, clf.n_classes)
    else:
        clf = CanViTForImageClassification.from_pretrained_with_probe(
            pretrained_repo=cfg.episode.model_repo, probe_repo=cfg.probe_repo,
            canvas_grid=FUSION_CANVAS_GRID,
        )
        log.info("Fused classifier from %s + %s (%d classes) using c=%d stats",
                 cfg.episode.model_repo, cfg.probe_repo, clf.n_classes, FUSION_CANVAS_GRID)
    return clf.to(device).eval()


@torch.inference_mode()
def evaluate(cfg: Config) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(cfg.device)

    canvas_grid = cfg.episode.canvas_grid
    assert canvas_grid is not None, "canvas_grid required for IN1k"

    require_existing_dir(cfg.val_dir, description="ImageNet validation directory", env_var="IMAGENET_VAL")
    clf = _load_classifier(cfg, device)

    patch_size_px = clf.canvit.backbone.patch_size_px
    assert cfg.scene_size % patch_size_px == 0, (
        f"scene_size={cfg.scene_size} must be divisible by patch_size_px={patch_size_px}"
    )
    img_size = cfg.scene_size
    dataset = make_in1k_dataset(cfg.val_dir, img_size)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)
    N, T = len(dataset), cfg.episode.n_timesteps

    all_top_k = torch.zeros(N, T, TOP_K, dtype=torch.int16)
    all_labels = torch.zeros(N, dtype=torch.int16)
    correct_top1 = torch.zeros(T, device=device, dtype=torch.long)
    t_start = time.monotonic()
    processed = 0

    for br in eval_batches(model=clf.canvit, loader=loader, episode_cfg=cfg.episode,
                           canvas_grid=canvas_grid, device=device, amp=cfg.amp):
        _, labels = br.batch
        labels_gpu = labels.to(device, non_blocking=True)
        B = labels.shape[0]
        all_labels[processed:processed + B] = labels.to(torch.int16)

        for step in br.steps:
            with torch.autocast(device_type=device.type, enabled=False):
                logits = clf.head(clf.norm(step.state.recurrent_cls[:, 0].float()))
            top_k = logits.topk(TOP_K, dim=-1).indices
            all_top_k[processed:processed + B, step.t] = top_k.cpu().to(torch.int16)
            correct_top1[step.t] += (top_k[:, 0] == labels_gpu).sum()
        processed += B

    for t in range(T):
        log.info("  t%d: top1=%.2f%%", t, 100 * correct_top1[t].item() / N)

    results = {
        "top_k_preds": all_top_k, "labels": all_labels,
        "metadata": {**collect_metadata(cfg), "wall_time_seconds": time.monotonic() - t_start, "n_images": N},
    }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, cfg.output)
    log.info("Saved to %s", cfg.output)
    return cfg.output
