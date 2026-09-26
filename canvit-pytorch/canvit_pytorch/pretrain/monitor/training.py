import torch
from torch import Tensor, nn


class MetricEMA:
    """Exponential moving averages of scalar metrics, kept on device until read."""

    def __init__(self, alpha: float) -> None:
        assert 0 < alpha <= 1, alpha
        self.alpha = alpha
        self.averages: dict[str, Tensor] = {}

    def update(self, metrics: dict[str, Tensor]) -> None:
        for name, value in metrics.items():
            value = value.detach().float()
            previous = self.averages.get(name)
            self.averages[name] = value if previous is None else self.alpha * value + (1 - self.alpha) * previous

    def read(self) -> dict[str, float]:
        """Synchronizes with the device."""
        return {name: value.item() for name, value in self.averages.items()}


def grad_norms_by_module(model: nn.Module, *, depth: int) -> dict[str, float]:
    """Gradient norm of each group of parameters sharing the first `depth` parts of their names."""
    groups: dict[str, list[Tensor]] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            groups.setdefault(".".join(name.split(".")[:depth]), []).append(parameter.grad.flatten())
    return {prefix: torch.cat(grads).norm().item() for prefix, grads in groups.items()}
