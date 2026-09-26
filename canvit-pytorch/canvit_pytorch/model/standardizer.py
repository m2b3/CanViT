import torch
from torch import Tensor, nn


class PositionAwareStandardizer(nn.Module):
    """Per-position z-scoring of [B, N, D] teacher features, fitted once on a sample and then fixed."""

    mean: Tensor  # [N, D]
    var: Tensor  # [N, D]
    fitted: Tensor  # scalar bool

    def __init__(self, n_positions: int, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(n_positions, dim))
        self.register_buffer("var", torch.ones(n_positions, dim))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, samples: Tensor) -> None:
        """Set the statistics from [S, N, D] samples."""
        assert samples.shape[1:] == self.mean.shape, (samples.shape, self.mean.shape)
        self.mean.copy_(samples.mean(dim=0))
        self.var.copy_(samples.var(dim=0, unbiased=False))
        self.fitted.fill_(True)

    @property
    def std(self) -> Tensor:
        return (self.var + self.eps).sqrt()

    def forward(self, x: Tensor) -> Tensor:
        assert bool(self.fitted), "Standardizer statistics were never fitted or loaded"
        return (x - self.mean) / self.std

    def destandardize(self, x: Tensor) -> Tensor:
        return x * self.std + self.mean
