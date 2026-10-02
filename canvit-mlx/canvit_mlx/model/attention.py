import mlx.core as mx
from canvit_core.config import CanvasProjections
from mlx import nn

from canvit_mlx.model.rope import RoPE, apply_2d_rope


class CanvasAttention(nn.Module):
    def __init__(
        self,
        *,
        query_dim: int,
        kv_dim: int,
        num_heads: int,
        q_map: nn.Module,
        k_map: nn.Module,
        v_map: nn.Module,
        o_map: nn.Module,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.ln_q = nn.LayerNorm(query_dim)
        self.ln_kv = nn.LayerNorm(kv_dim)
        self.q_map, self.k_map, self.v_map, self.o_map = q_map, k_map, v_map, o_map

    def __call__(self, *, query: mx.array, kv: mx.array, query_rope: RoPE, kv_rope: RoPE) -> mx.array:
        def multihead(x: mx.array) -> mx.array:
            return x.reshape(*x.shape[:2], self.num_heads, -1).transpose(0, 2, 1, 3)

        query = apply_2d_rope(multihead(self.q_map(self.ln_q(query))), query_rope)
        kv = self.ln_kv(kv)
        key = apply_2d_rope(multihead(self.k_map(kv)), kv_rope)
        value = multihead(self.v_map(kv))
        output = mx.fast.scaled_dot_product_attention(
            query,
            key.astype(query.dtype),
            value.astype(query.dtype),
            scale=query.shape[-1] ** -0.5,
        )
        return self.o_map(output.transpose(0, 2, 1, 3).reshape(output.shape[0], output.shape[2], -1))


class CanvasAttentionRead(CanvasAttention):
    def __init__(self, *, backbone_dim: int, canvas_dim: int, num_heads: int, projections: CanvasProjections) -> None:
        super().__init__(
            query_dim=backbone_dim,
            kv_dim=canvas_dim,
            num_heads=num_heads,
            q_map=nn.Linear(backbone_dim, canvas_dim),
            o_map=nn.Linear(canvas_dim, backbone_dim),
            k_map=nn.Linear(canvas_dim, canvas_dim) if projections == "qkvo" else nn.Identity(),
            v_map=nn.Linear(canvas_dim, canvas_dim) if projections == "qkvo" else nn.Identity(),
        )


class CanvasAttentionWrite(CanvasAttention):
    def __init__(self, *, backbone_dim: int, canvas_dim: int, num_heads: int, projections: CanvasProjections) -> None:
        super().__init__(
            query_dim=canvas_dim,
            kv_dim=backbone_dim,
            num_heads=num_heads,
            k_map=nn.Linear(backbone_dim, canvas_dim),
            v_map=nn.Linear(backbone_dim, canvas_dim),
            q_map=nn.Linear(canvas_dim, canvas_dim) if projections == "qkvo" else nn.Identity(),
            o_map=nn.Linear(canvas_dim, canvas_dim) if projections == "qkvo" else nn.Identity(),
        )
