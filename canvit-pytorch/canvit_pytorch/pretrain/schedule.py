from torch.optim import Optimizer
from torch.optim.lr_scheduler import ConstantLR, LinearLR, LRScheduler, SequentialLR


def warmup_then_constant(optimizer: Optimizer, *, start_lr: float, peak_lr: float, warmup_steps: int) -> LRScheduler:
    """Linear warmup from start_lr to peak_lr over warmup_steps, then peak_lr; the optimizer's lr must be peak_lr."""
    assert warmup_steps >= 1 and 0 < start_lr <= peak_lr, (warmup_steps, start_lr, peak_lr)
    assert all(group["lr"] == peak_lr for group in optimizer.param_groups), "construct the optimizer with lr=peak_lr"
    warmup = LinearLR(optimizer, start_factor=start_lr / peak_lr, end_factor=1.0, total_iters=warmup_steps)
    constant = ConstantLR(optimizer, factor=1.0, total_iters=0)
    return SequentialLR(optimizer, schedulers=[warmup, constant], milestones=[warmup_steps])
