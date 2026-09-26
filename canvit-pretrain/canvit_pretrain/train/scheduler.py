"""Learning rate scheduler utilities."""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)


def _make_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    peak_lr: float,
    start_lr: float | None,
) -> LinearLR:
    effective_start_lr = start_lr if start_lr is not None else peak_lr / warmup_steps
    return LinearLR(
        optimizer,
        start_factor=effective_start_lr / peak_lr,
        end_factor=1.0,
        total_iters=warmup_steps,
    )


def warmup_constant_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    peak_lr: float,
    start_lr: float | None = None,
) -> LRScheduler:
    """Warmup → constant at peak_lr."""
    assert warmup_steps >= 0
    if warmup_steps == 0:
        return ConstantLR(optimizer, factor=1.0, total_iters=0)

    warmup = _make_warmup(optimizer, warmup_steps, peak_lr, start_lr)
    constant = ConstantLR(optimizer, factor=1.0, total_iters=0)
    return SequentialLR(optimizer, schedulers=[warmup, constant], milestones=[warmup_steps])


def warmup_cosine_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    peak_lr: float,
    start_lr: float | None = None,
    min_lr: float = 0.0,
) -> LRScheduler:
    """Warmup → cosine decay from peak_lr to min_lr over total_steps."""
    assert warmup_steps >= 0
    assert total_steps > warmup_steps, f"total_steps ({total_steps}) must exceed warmup_steps ({warmup_steps})"

    if warmup_steps == 0:
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)

    warmup = _make_warmup(optimizer, warmup_steps, peak_lr, start_lr)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=min_lr)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
