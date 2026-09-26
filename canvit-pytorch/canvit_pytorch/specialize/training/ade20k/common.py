"""Shared training infrastructure for ADE20K probe training."""

import torch
from dinov3.eval.segmentation.schedulers import WarmupOneCycleLR
from dinov3.eval.segmentation.transforms import make_segmentation_train_transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from canvit_specialize.datasets.ade20k import ADE20kDataset, make_val_transforms
from canvit_specialize.training.ade20k.config import ProbeTrainBase


def make_ade20k_loaders(cfg: ProbeTrainBase) -> tuple[DataLoader, DataLoader]:
    """Build ADE20K train/val data loaders."""
    if not cfg.ade20k_root.exists():
        raise FileNotFoundError(
            f"ADE20K root not found: {cfg.ade20k_root}. "
            "Set ADE20K_ROOT or pass --ade20k-root."
        )

    _train_aug = make_segmentation_train_transforms(
        img_size=cfg.scene_size,
        random_img_size_ratio_range=list(cfg.aug_scale_range),
        # Upstream annotation is Tuple[int] but implementation expects (H, W).
        crop_size=(cfg.scene_size, cfg.scene_size),  # pyright: ignore[reportArgumentType]
        flip_prob=cfg.aug_flip_prob,
        reduce_zero_label=True,
    )

    def train_transform(img, mask):
        img_t, mask_t = _train_aug(img, mask)
        return img_t, mask_t.squeeze(0)

    train_ds = ADE20kDataset(root=cfg.ade20k_root, split="training", joint_transform=train_transform)
    val_img_tf, val_mask_tf = make_val_transforms(cfg.scene_size, "squish")
    val_ds = ADE20kDataset(root=cfg.ade20k_root, split="validation", img_transform=val_img_tf, mask_transform=val_mask_tf)

    train_loader = DataLoader(
        train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(val_ds, cfg.eval_batch_size, num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader


def make_optimizer_and_scheduler(
    params, *, lr: float, weight_decay: float, max_steps: int,
    warmup_steps: int, warmup_lr_ratio: float,
) -> tuple[AdamW, LRScheduler]:
    """Create AdamW + WarmupOneCycleLR."""
    optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = WarmupOneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=max_steps,
        warmup_iters=warmup_steps,
        warmup_ratio=warmup_lr_ratio,
        pct_start=0,
        anneal_strategy="cos",
        final_div_factor=float("inf"),
        use_beta1=False,
        update_momentum=False,
    )
    return optimizer, scheduler


def make_amp_ctx(amp: bool, device: torch.device) -> torch.autocast:
    """Create autocast context for mixed-precision training."""
    amp_dtype = torch.bfloat16 if amp else torch.float32
    return torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp)
