from torch import Tensor, nn
from torch.nn import functional as F

from canvit_pytorch.model.rope import RoPE, apply_2d_rope


def to_multihead(x: Tensor, num_heads: int) -> Tensor:
    """[B, N, D] -> [B, heads, N, D / heads]."""
    B, N, D = x.shape
    return x.view(B, N, num_heads, D // num_heads).transpose(1, 2)


def from_multihead(x: Tensor) -> Tensor:
    """[B, heads, N, head_dim] -> [B, N, heads * head_dim]."""
    B, H, N, head_dim = x.shape
    return x.transpose(1, 2).reshape(B, N, H * head_dim)


class CanvasAttention(nn.Module):
    """Cross-attention between the glimpse stream and the canvas, computed at the canvas width.

    Both inputs are layer-normalized and rotated with SR-RoPE. Subclasses choose
    which of the query, key, value and output projections are learned; the
    others stay identities.
    """

    def __init__(self, *, query_dim: int, kv_dim: int, canvas_dim: int, num_heads: int) -> None:
        super().__init__()
        assert canvas_dim % num_heads == 0, (canvas_dim, num_heads)
        self.num_heads = num_heads
        self.q_map: nn.Module = nn.Identity()
        self.k_map: nn.Module = nn.Identity()
        self.v_map: nn.Module = nn.Identity()
        self.o_map: nn.Module = nn.Identity()
        self.ln_q = nn.LayerNorm(query_dim)
        self.ln_kv = nn.LayerNorm(kv_dim)

    def forward(self, *, query: Tensor, kv: Tensor, query_rope: RoPE, kv_rope: RoPE) -> Tensor:
        """[B, N_query, query_dim] attends to [B, N_kv, kv_dim]; returns the residual for the query stream."""
        q = apply_2d_rope(to_multihead(self.q_map(self.ln_q(query)), self.num_heads), query_rope)
        kv = self.ln_kv(kv)
        k = apply_2d_rope(to_multihead(self.k_map(kv), self.num_heads), kv_rope)
        v = to_multihead(self.v_map(kv), self.num_heads)
        # Under autocast the glimpse side runs in bfloat16 while float32 canvas keys and values
        # bypass any projection; SDPA needs one dtype.
        out = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype))
        return self.o_map(from_multihead(out))
