import mlx.core as mx
from mlx import nn

from canvit_mlx.module import Module


class PositionAwareStandardizer(Module):
    def __init__(self, n_positions: int, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.mean = mx.zeros((n_positions, dim))
        self.var = mx.ones((n_positions, dim))
        self.fitted = mx.array(False)
        self.freeze()

    @staticmethod
    def trainable_parameter_filter(module: nn.Module, key: str, value: object) -> bool:
        return Module.trainable_parameter_filter(module, key, value) and key not in {"mean", "var", "fitted"}

    def fit(self, samples: mx.array) -> None:
        assert samples.shape[1:] == self.mean.shape, (samples.shape, self.mean.shape)
        self.mean = mx.mean(samples, axis=0)
        self.var = mx.var(samples, axis=0)
        self.fitted = mx.array(True)

    @property
    def std(self) -> mx.array:
        return mx.sqrt(self.var + self.eps)

    def __call__(self, x: mx.array) -> mx.array:
        assert bool(self.fitted), "Standardizer statistics were never fitted or loaded"
        return (x - self.mean) / self.std

    def destandardize(self, x: mx.array) -> mx.array:
        return x * self.std + self.mean
