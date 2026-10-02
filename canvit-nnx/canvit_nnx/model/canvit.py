"""CanViT's dual-stream backbone and recurrent canvas state."""

import math
from typing import NamedTuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from canvit_core.attention import canvas_attention_schedule
from canvit_core.config import CanViTConfig

from canvit_nnx.model.attention import CanvasAttentionRead, CanvasAttentionWrite
from canvit_nnx.model.backbone import ViTBackbone
from canvit_nnx.model.rope import compute_2d_rope, make_rope_periods
from canvit_nnx.model.vpe import ViewpointEncoding
from canvit_nnx.viewpoint import Viewpoint, grid_coords, viewpoint_grid_coords

Array = jax.Array


class RecurrentState(NamedTuple):
    canvas: Array
    recurrent_cls: Array

    def detach(self) -> "RecurrentState":
        return RecurrentState(
            canvas=jax.lax.stop_gradient(self.canvas),
            recurrent_cls=jax.lax.stop_gradient(self.recurrent_cls),
        )


class CanViTOutput(NamedTuple):
    state: RecurrentState
    glimpse_patches: Array
    vpe: Array | None


class CanViT(nnx.Module):
    def __init__(self, config: CanViTConfig, *, rngs: nnx.Rngs) -> None:
        self.config = config
        spec = config.backbone_spec
        self.backbone = ViTBackbone(spec=spec, rngs=rngs)
        self.read_after_blocks, self.write_after_blocks = canvas_attention_schedule(
            num_blocks=spec.num_blocks,
            rw_stride=config.rw_stride,
            enable_reads=config.enable_reads,
        )
        self.canvas_reads = nnx.List(
            [
                CanvasAttentionRead(
                    backbone_dim=spec.embed_dim,
                    canvas_dim=config.canvas_dim,
                    num_heads=config.canvas_num_heads,
                    projections=config.canvas_projections,
                    rngs=rngs,
                )
                for _ in self.read_after_blocks
            ]
        )
        self.canvas_writes = nnx.List(
            [
                CanvasAttentionWrite(
                    backbone_dim=spec.embed_dim,
                    canvas_dim=config.canvas_dim,
                    num_heads=config.canvas_num_heads,
                    projections=config.canvas_projections,
                    rngs=rngs,
                )
                for _ in self.write_after_blocks
            ]
        )

        canvas_scale = 1.0 / math.sqrt(config.canvas_dim)
        self.init_canvas_registers = nnx.Param(
            jax.random.normal(rngs.params(), (1, config.num_canvas_registers, config.canvas_dim)) * canvas_scale
        )
        self.init_canvas_patch = nnx.Param(jax.random.normal(rngs.params(), (1, 1, config.canvas_dim)) * canvas_scale)
        self.init_recurrent_cls = nnx.Param(
            jax.random.normal(rngs.params(), (1, 1, spec.embed_dim)) / math.sqrt(spec.embed_dim)
        )
        self.backbone_registers = nnx.Param(
            jax.random.normal(rngs.params(), (1, config.num_backbone_registers, spec.embed_dim)) * 0.02
        )
        self.vpe = ViewpointEncoding(dim=spec.embed_dim, rngs=rngs) if config.enable_vpe else None

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
        registers = jnp.broadcast_to(
            self.init_canvas_registers[...],
            (batch_size, self.config.num_canvas_registers, self.canvas_dim),
        )
        patches = jnp.broadcast_to(
            self.init_canvas_patch[...],
            (batch_size, canvas_grid_size**2, self.canvas_dim),
        )
        recurrent_cls = jnp.broadcast_to(
            self.init_recurrent_cls[...],
            (batch_size, 1, self.backbone_dim),
        )
        return RecurrentState(
            canvas=jnp.concatenate((registers, patches), axis=1),
            recurrent_cls=recurrent_cls,
        )

    def canvas_patches(self, canvas: Array) -> Array:
        assert canvas.ndim == 3 and canvas.shape[-1] == self.canvas_dim, canvas.shape
        assert canvas.shape[1] >= self.config.num_canvas_registers, canvas.shape
        return canvas[:, self.config.num_canvas_registers :, :]

    def canvas_patch_grid(self, canvas: Array) -> Array:
        patches = self.canvas_patches(canvas)
        grid_size = math.isqrt(int(patches.shape[1]))
        assert grid_size * grid_size == patches.shape[1], (
            patches.shape,
            "canvas patches do not form a square grid",
        )
        return patches.reshape(patches.shape[0], grid_size, grid_size, self.canvas_dim)

    def __call__(self, *, glimpse: Array, state: RecurrentState, viewpoint: Viewpoint) -> CanViTOutput:
        assert glimpse.ndim == 4, glimpse.shape
        batch_size, height, width, _ = glimpse.shape
        assert height == width and height % self.patch_size == 0, (
            "glimpse must be square in whole patches",
            glimpse.shape,
        )
        assert state.canvas.shape[0] == batch_size == viewpoint.scales.shape[0], (
            glimpse.shape,
            state.canvas.shape,
            viewpoint.scales.shape,
        )
        assert state.canvas.ndim == 3 and state.canvas.shape[-1] == self.canvas_dim, state.canvas.shape
        assert state.recurrent_cls.shape == (batch_size, 1, self.backbone_dim), state.recurrent_cls.shape
        glimpse_grid_size = height // self.patch_size
        patches = self.backbone.patch_embed(glimpse)

        tokens_before_patches: list[Array] = []
        if self.config.enable_vpe:
            assert self.vpe is not None
            tokens_before_patches.append(self.vpe(viewpoint)[:, None, :].astype(patches.dtype))
        tokens_before_patches.extend(
            (
                state.recurrent_cls,
                jnp.broadcast_to(
                    self.backbone_registers[...],
                    (batch_size, self.config.num_backbone_registers, self.backbone_dim),
                ),
            )
        )
        num_prefix_tokens = sum(token.shape[1] for token in tokens_before_patches)
        tokens = jnp.concatenate((*tokens_before_patches, patches), axis=1)

        spec = self.config.backbone_spec
        backbone_periods = make_rope_periods(head_dim=spec.head_dim, base=spec.rope_base)
        canvas_periods = make_rope_periods(head_dim=self.config.canvas_head_dim, base=spec.rope_base)
        glimpse_positions = viewpoint_grid_coords(viewpoint, size=glimpse_grid_size).reshape(batch_size, -1, 2)
        glimpse_rope = compute_2d_rope(positions=glimpse_positions, periods=backbone_periods)
        glimpse_canvas_rope = compute_2d_rope(positions=glimpse_positions, periods=canvas_periods)

        canvas = state.canvas
        canvas_grid_size = math.isqrt(int(canvas.shape[1] - self.config.num_canvas_registers))
        assert canvas_grid_size**2 == canvas.shape[1] - self.config.num_canvas_registers, canvas.shape
        canvas_positions = grid_coords(size=canvas_grid_size).reshape(1, -1, 2)
        canvas_rope = compute_2d_rope(positions=canvas_positions, periods=canvas_periods)

        reads = dict(zip(self.read_after_blocks, self.canvas_reads, strict=True))
        writes = dict(zip(self.write_after_blocks, self.canvas_writes, strict=True))
        for block_index, block in enumerate(self.backbone.blocks):
            tokens = block(tokens, glimpse_rope)
            if (read := reads.get(block_index)) is not None:
                tokens = tokens + read(
                    query=tokens,
                    kv=canvas,
                    query_rope=glimpse_canvas_rope,
                    kv_rope=canvas_rope,
                )
            if (write := writes.get(block_index)) is not None:
                canvas = canvas + write(
                    query=canvas,
                    kv=tokens,
                    query_rope=canvas_rope,
                    kv_rope=glimpse_canvas_rope,
                )

        cls_index = 1 if self.config.enable_vpe else 0
        return CanViTOutput(
            state=RecurrentState(
                canvas=canvas,
                recurrent_cls=tokens[:, cls_index : cls_index + 1, :],
            ),
            glimpse_patches=tokens[:, num_prefix_tokens:, :],
            vpe=tokens[:, 0, :] if self.config.enable_vpe else None,
        )
