import mlx.core as mx
from mlx import nn


class LinearReadout(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(self.norm(x))
