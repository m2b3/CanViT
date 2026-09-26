"""Full-QKVO canvas attention variants (ablation).

Dense Linear projections on BOTH local and canvas streams,
unlike the default asymmetric attention (Identity on canvas side).
"""

from typing import override

import torch.nn.functional as F
from torch import Tensor, nn

from canvit_pytorch.attention.base import CanvasAttention, from_multihead, to_multihead
from canvit_pytorch.rope import RoPE, rope_apply_with_prefix


class CanvasReadAttentionFull(CanvasAttention):
    """Local queries canvas — all four projections dense.

    vs CanvasReadAttention: adds K, V projections on canvas side.
    """

    def __init__(
        self,
        *,
        local_dim: int,
        canvas_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__(
            q_in_dim=local_dim,
            kv_in_dim=canvas_dim,
            canvas_dim=canvas_dim,
            out_dim=local_dim,
            num_heads=num_heads,
        )
        self.q_proj = nn.Linear(local_dim, canvas_dim)
        self.k_proj = nn.Linear(canvas_dim, canvas_dim)
        self.v_proj = nn.Linear(canvas_dim, canvas_dim)
        self.out_proj = nn.Linear(canvas_dim, local_dim)


class CanvasWriteAttentionFull(CanvasAttention):
    """Canvas queries local — all four projections dense.

    vs CanvasWriteAttention: adds Q, O projections on canvas side.
    """

    def __init__(
        self,
        *,
        local_dim: int,
        canvas_dim: int,
        num_heads: int,
        gate_bias_init: float | None = None,
    ) -> None:
        super().__init__(
            q_in_dim=canvas_dim,
            kv_in_dim=local_dim,
            canvas_dim=canvas_dim,
            out_dim=canvas_dim,
            num_heads=num_heads,
        )
        self.q_proj = nn.Linear(canvas_dim, canvas_dim)
        self.k_proj = nn.Linear(local_dim, canvas_dim)
        self.v_proj = nn.Linear(local_dim, canvas_dim)
        self.out_proj = nn.Linear(canvas_dim, canvas_dim)

        self.gate_linear: nn.Linear | None = None
        if gate_bias_init is not None:
            self.gate_linear = nn.Linear(canvas_dim, 1)
            nn.init.constant_(self.gate_linear.bias, gate_bias_init)

    @property
    def is_convex(self) -> bool:
        return self.gate_linear is not None

    @override
    def forward(
        self,
        *,
        query: Tensor,
        kv: Tensor,
        query_rope: RoPE,
        kv_rope: RoPE,
    ) -> Tensor:
        if self.gate_linear is None:
            return super().forward(query=query, kv=kv, query_rope=query_rope, kv_rope=kv_rope)

        q = to_multihead(self.q_proj(self.q_norm(query)), self.num_heads)
        kv_normed = self.kv_norm(kv)
        k = to_multihead(self.k_proj(kv_normed), self.num_heads)
        v = to_multihead(self.v_proj(kv_normed), self.num_heads)

        q = rope_apply_with_prefix(x=q, rope=query_rope)
        k = rope_apply_with_prefix(x=k, rope=kv_rope)

        attn_out = self.out_proj(from_multihead(F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype))))
        gate = self.gate_linear(attn_out).sigmoid()
        return query.lerp(attn_out.to(query.dtype), gate.to(query.dtype))
