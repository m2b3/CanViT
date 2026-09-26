from torch import Tensor, nn


class LinearReadout(nn.Module):
    """LayerNorm then a linear map, applied token-wise: how CanViT's tokens are decoded (paper, Eq. 1)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.norm(x))
