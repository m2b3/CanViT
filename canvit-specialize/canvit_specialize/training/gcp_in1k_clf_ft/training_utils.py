"""Pure utilities for training — no XLA dependency, fully testable on CPU."""

import logging
import math
import os
import time
from pathlib import Path
from typing import NamedTuple

import torch

log = logging.getLogger("training_utils")


class ValLoader(NamedTuple):
    loader: torch.utils.data.DataLoader
    n_glimpses: int
    tag: str      # metric key prefix (e.g., "c2f"). "" = primary.
    display: str  # human-readable label for stdout


def to_cpu(obj):
    """Recursively move tensors in nested dicts/lists to CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_cpu(v) for v in obj]
    return obj


def save_checkpoint(*, checkpoint_dir: str, step: int, clf, optimizer,
                    best_val_acc: float, comet_key: str | None,
                    filename: str = "latest.pt", sync_fn=None) -> str:
    """Atomic checkpoint save. Returns path written. Logs per-phase wallclock."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    tmp = os.path.join(checkpoint_dir, f".tmp_{filename}_{os.getpid()}_{step}")
    t_sync0 = time.perf_counter()
    if sync_fn is not None:
        sync_fn()
    t_state0 = time.perf_counter()
    state = to_cpu({
        "step": step, "best_val_acc": best_val_acc, "comet_key": comet_key,
        "model_state_dict": clf.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    })
    t_write0 = time.perf_counter()
    torch.save(state, tmp)
    os.replace(tmp, path)
    t_end = time.perf_counter()
    size_mb = os.path.getsize(path) / 1e6
    log.info("  Checkpoint saved: %s (step %d) [%.1fMB | sync %.2fs + state %.2fs + write %.2fs = %.2fs]",
             path, step, size_mb,
             t_state0 - t_sync0, t_write0 - t_state0, t_end - t_write0, t_end - t_sync0)
    return path


def load_checkpoint(path: str | Path) -> dict:
    """Load checkpoint from file."""
    log.info("Loading checkpoint from %s...", path)
    return torch.load(path, map_location="cpu", weights_only=False)


def apply_model_weights(state: dict, clf, device) -> None:
    """Load model weights from checkpoint state dict onto device."""
    clf.load_state_dict({k: v.to(device) for k, v in state["model_state_dict"].items()})


def maybe_resume(*, checkpoint_dir: str, clf, optimizer, device) -> tuple[int, float, str | None]:
    """Resume from latest.pt if it exists. Returns (start_step, best_val_acc, comet_key)."""
    path = Path(checkpoint_dir) / "latest.pt"
    if not path.exists():
        return 0, -1.0, None
    state = load_checkpoint(path)
    apply_model_weights(state, clf, device)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    start_step = state["step"]
    best_val_acc = state.get("best_val_acc", -1.0)
    comet_key = state.get("comet_key")
    log.info("Resumed at step %d, best_val_acc=%.4f, comet=%s", start_step, best_val_acc, comet_key or 'none')
    return start_step, best_val_acc, comet_key


def make_lr_lambda(*, warmup_steps: int, total_steps: int):
    """Linear warmup → cosine decay. Returns lambda for LambdaLR."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def should_early_stop(*, val_acc: float, best_val_acc: float, delta: float | None) -> bool:
    """Returns True if val_acc has regressed more than delta below best."""
    if delta is None:
        return False
    return best_val_acc - val_acc > delta
