"""Training state for probes."""

from dataclasses import dataclass
from typing import Any

from torch import Tensor
from torch.optim import AdamW

from canvit_pytorch.probes import SegmentationProbe


@dataclass
class ProbeState:
    """Training state for one probe."""

    name: str
    head: SegmentationProbe
    optimizer: AdamW
    scheduler: Any  # WarmupOneCycleLR or other LR scheduler
    n_timesteps: int = 0
    _best_mious: list[float] | None = None
    _loss_sum: Tensor | None = None
    _grad_norm_sum: Tensor | None = None
    _count: int = 0

    def init_best_mious(self, n_timesteps: int) -> None:
        self.n_timesteps = n_timesteps
        self._best_mious = [0.0] * n_timesteps

    @property
    def best_mious(self) -> list[float]:
        assert self._best_mious is not None, "call init_best_mious first"
        return self._best_mious

    @property
    def best_last_miou(self) -> float:
        return self.best_mious[-1]

    def update_best(self, mious: list[float]) -> bool:
        """Update per-timestep bests. Returns True if last timestep improved."""
        assert len(mious) == self.n_timesteps
        old_last = self.best_last_miou
        for t, v in enumerate(mious):
            if v > self.best_mious[t]:
                self.best_mious[t] = v
        return self.best_last_miou > old_last

    def accumulate(self, loss: Tensor, grad_norm: Tensor) -> None:
        """Accumulate loss/grad_norm. NO GPU sync."""
        if self._loss_sum is None:
            self._loss_sum = loss.detach().clone()
            self._grad_norm_sum = grad_norm.detach().clone()
        else:
            self._loss_sum += loss.detach()
            assert self._grad_norm_sum is not None
            self._grad_norm_sum += grad_norm.detach()
        self._count += 1

    def get_and_reset(self) -> tuple[float, float]:
        """Get averaged stats and reset. SYNCS here."""
        assert self._loss_sum is not None and self._grad_norm_sum is not None
        avg_loss = (self._loss_sum / self._count).item()
        avg_grad = (self._grad_norm_sum / self._count).item()
        self._loss_sum = self._grad_norm_sum = None
        self._count = 0
        return avg_loss, avg_grad
