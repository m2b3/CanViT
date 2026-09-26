"""Fine-tuning checkpoints: classifier and optimizer state, step, best validation accuracy, Comet experiment key."""

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import torch

from canvit_pytorch import CanViTForImageClassification

log = logging.getLogger(__name__)

LATEST = "latest.pt"


class Progress(NamedTuple):
    step: int
    best_val_acc: float  # -1 before the first validation
    comet_key: str | None


def to_cpu(tree: Any) -> Any:
    """Tensors of nested dicts and lists, moved to the CPU."""
    if isinstance(tree, torch.Tensor):
        return tree.cpu()
    if isinstance(tree, dict):
        return {key: to_cpu(value) for key, value in tree.items()}
    if isinstance(tree, list):
        return [to_cpu(value) for value in tree]
    return tree


def save_checkpoint(
    *, checkpoint_dir: Path, filename: str, progress: Progress, classifier: CanViTForImageClassification,
    optimizer: torch.optim.Optimizer, wait_for_device: Callable[[], None],
) -> Path:
    """Write atomically, logging the time spent waiting for the device, gathering state and writing."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / filename
    staged = checkpoint_dir / f".tmp_{filename}_{os.getpid()}_{progress.step}"
    start = time.perf_counter()
    wait_for_device()
    waited = time.perf_counter()
    state = to_cpu({
        "step": progress.step, "best_val_acc": progress.best_val_acc, "comet_key": progress.comet_key,
        "model_state_dict": classifier.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    })
    gathered = time.perf_counter()
    torch.save(state, staged)
    os.replace(staged, path)
    written = time.perf_counter()
    log.info("  Checkpoint saved: %s (step %d) [%.1fMB | sync %.2fs + state %.2fs + write %.2fs = %.2fs]",
             path, progress.step, path.stat().st_size / 1e6,
             waited - start, gathered - waited, written - gathered, written - start)
    return path


def load_checkpoint(path: Path) -> dict[str, Any]:
    log.info("Loading checkpoint from %s...", path)
    return torch.load(path, map_location="cpu", weights_only=False)


def load_classifier_weights(state: dict[str, Any], classifier: CanViTForImageClassification, device: torch.device) -> None:
    classifier.load_state_dict({key: value.to(device) for key, value in state["model_state_dict"].items()})


def resume(
    *, checkpoint_dir: Path, classifier: CanViTForImageClassification, optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Progress:
    """Restore classifier and optimizer from checkpoint_dir/latest.pt when it exists."""
    path = checkpoint_dir / LATEST
    if not path.exists():
        return Progress(step=0, best_val_acc=-1.0, comet_key=None)
    state = load_checkpoint(path)
    load_classifier_weights(state, classifier, device)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    progress = Progress(step=state["step"], best_val_acc=state["best_val_acc"], comet_key=state["comet_key"])
    log.info("Resumed at step %d, best_val_acc=%.4f, comet=%s", progress.step, progress.best_val_acc, progress.comet_key)
    return progress
