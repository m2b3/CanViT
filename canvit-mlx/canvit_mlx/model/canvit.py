import math
from typing import NamedTuple

import mlx.core as mx
from canvit_core import CanViTConfig
from canvit_core.attention import canvas_attention_schedule

from canvit_mlx.model.attention import CanvasAttentionRead, CanvasAttentionWrite
from canvit_mlx.model.backbone import ViTBackbone
from canvit_mlx.model.rope import compute_2d_rope
from canvit_mlx.model.vpe import ViewpointEncoding
from canvit_mlx.module import Module
from canvit_mlx.viewpoint import Viewpoint, grid_coords, viewpoint_grid_coords


class RecurrentState(NamedTuple):
    canvas: mx.array  # [B, num_canvas_registers + G*G, canvas_dim]
    recurrent_cls: mx.array  # [B, 1, backbone_dim]

    def detach(self) -> "RecurrentState":
        return RecurrentState(mx.stop_gradient(self.canvas), mx.stop_gradient(self.recurrent_cls))


class CanViTOutput(NamedTuple):
    state: RecurrentState
    glimpse_patches: mx.array  # [B, g*g, backbone_dim]
    vpe: mx.array | None  # [B, backbone_dim]


class CanViT(Module):
    def __init__(self, config: CanViTConfig) -> None:
        super().__init__()
        self.config = config
        spec = config.backbone_spec
        self.backbone = ViTBackbone(spec)
        self.read_after_blocks, self.write_after_blocks = canvas_attention_schedule(
            num_blocks=spec.num_blocks,
            rw_stride=config.rw_stride,
            enable_reads=config.enable_reads,
        )
        dimensions = dict(backbone_dim=spec.embed_dim, canvas_dim=config.canvas_dim, num_heads=config.canvas_num_heads)
        self.canvas_reads = [
            CanvasAttentionRead(**dimensions, projections=config.canvas_projections) for _ in self.read_after_blocks
        ]
        self.canvas_writes = [
            CanvasAttentionWrite(**dimensions, projections=config.canvas_projections) for _ in self.write_after_blocks
        ]
        self.init_canvas_registers = mx.random.normal((1, config.num_canvas_registers, config.canvas_dim)) / math.sqrt(
            config.canvas_dim
        )
        self.init_canvas_patch = mx.random.normal((1, 1, config.canvas_dim)) / math.sqrt(config.canvas_dim)
        self.init_recurrent_cls = mx.random.normal((1, 1, spec.embed_dim)) / math.sqrt(spec.embed_dim)
        self.backbone_registers = mx.random.normal((1, config.num_backbone_registers, spec.embed_dim)) * 0.02
        self.vpe = ViewpointEncoding(spec.embed_dim) if config.enable_vpe else None

    @property
    def backbone_dim(self) -> int:
        return self.config.backbone_spec.embed_dim

    @property
    def canvas_dim(self) -> int:
        return self.config.canvas_dim

    @property
    def patch_size(self) -> int:
        return self.config.backbone_spec.patch_size

    def init_state(self, *, batch_size: int, canvas_grid_size: int) -> RecurrentState:
        assert batch_size > 0 and canvas_grid_size > 0, (batch_size, canvas_grid_size)
        registers = mx.broadcast_to(
            self.init_canvas_registers, (batch_size, self.config.num_canvas_registers, self.canvas_dim)
        )
        patches = mx.broadcast_to(self.init_canvas_patch, (batch_size, canvas_grid_size**2, self.canvas_dim))
        return RecurrentState(
            canvas=mx.concatenate([registers, patches], axis=1),
            recurrent_cls=mx.broadcast_to(self.init_recurrent_cls, (batch_size, 1, self.backbone_dim)),
        )

    def canvas_patches(self, canvas: mx.array) -> mx.array:
        return canvas[:, self.config.num_canvas_registers :]

    def canvas_patch_grid(self, canvas: mx.array) -> mx.array:
        patches = self.canvas_patches(canvas)
        grid = math.isqrt(patches.shape[1])
        assert grid > 0 and grid * grid == patches.shape[1], patches.shape
        return patches.reshape(patches.shape[0], grid, grid, self.canvas_dim)

    def __call__(self, *, glimpse: mx.array, state: RecurrentState, viewpoint: Viewpoint) -> CanViTOutput:
        """Process [B, px, px, 3] glimpses; carry the returned state into the next call."""
        batch_size, height, width, channels = glimpse.shape
        assert channels == 3 and height == width and height % self.patch_size == 0, glimpse.shape
        assert state.recurrent_cls.shape == (batch_size, 1, self.backbone_dim), state.recurrent_cls.shape
        assert state.canvas.ndim == 3, state.canvas.shape
        assert state.canvas.shape[0] == batch_size and state.canvas.shape[2] == self.canvas_dim, state.canvas.shape
        canvas_grid = self.canvas_patch_grid(state.canvas).shape[1]
        glimpse_grid = height // self.patch_size
        patches = self.backbone.patch_embed(glimpse)
        prefix = [
            state.recurrent_cls,
            mx.broadcast_to(
                self.backbone_registers, (batch_size, self.config.num_backbone_registers, self.backbone_dim)
            ),
        ]
        if self.vpe is not None:
            prefix.insert(0, self.vpe(viewpoint)[:, None].astype(patches.dtype))
        prefix_size = sum(tokens.shape[1] for tokens in prefix)
        tokens = mx.concatenate([*prefix, patches], axis=1)
        spec = self.config.backbone_spec
        positions = viewpoint_grid_coords(viewpoint, size=glimpse_grid).reshape(batch_size, -1, 2)
        glimpse_rope = compute_2d_rope(positions=positions, head_dim=spec.head_dim, base=spec.rope_base)
        glimpse_canvas_rope = compute_2d_rope(
            positions=positions, head_dim=self.config.canvas_head_dim, base=spec.rope_base
        )
        canvas_rope = compute_2d_rope(
            positions=grid_coords(size=canvas_grid).reshape(1, -1, 2),
            head_dim=self.config.canvas_head_dim,
            base=spec.rope_base,
        )
        canvas = state.canvas
        reads = dict(zip(self.read_after_blocks, self.canvas_reads, strict=True))
        writes = dict(zip(self.write_after_blocks, self.canvas_writes, strict=True))
        for index, block in enumerate(self.backbone.blocks):
            tokens = block(tokens, glimpse_rope)
            if index in reads:
                tokens = tokens + reads[index](
                    query=tokens, kv=canvas, query_rope=glimpse_canvas_rope, kv_rope=canvas_rope
                )
            if index in writes:
                canvas = canvas + writes[index](
                    query=canvas, kv=tokens, query_rope=canvas_rope, kv_rope=glimpse_canvas_rope
                )
        cls_index = int(self.vpe is not None)
        return CanViTOutput(
            state=RecurrentState(canvas=canvas, recurrent_cls=tokens[:, cls_index : cls_index + 1]),
            glimpse_patches=tokens[:, prefix_size:],
            vpe=tokens[:, 0] if self.vpe is not None else None,
        )
