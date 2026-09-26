"""CanViT: a ViT backbone that reads from and writes to a scene-wide canvas, one glimpse at a time."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from canvit_pytorch.model.attention import CANVAS_ATTENTION_CLASSES
from canvit_pytorch.model.backbone import ViTBackbone
from canvit_pytorch.model.config import CanViTConfig
from canvit_pytorch.model.rope import compute_2d_rope, make_rope_periods
from canvit_pytorch.model.vpe import ViewpointEncoding
from canvit_pytorch.viewpoint import Viewpoint, grid_coords, viewpoint_grid_coords


@dataclass(frozen=True)
class RecurrentState:
    """What CanViT carries from one glimpse to the next."""

    canvas: Tensor  # [B, num_canvas_registers + G*G, canvas_dim]: registers, then a G×G grid, row-major
    recurrent_cls: Tensor  # [B, 1, backbone_dim]

    def detach(self) -> "RecurrentState":
        return RecurrentState(canvas=self.canvas.detach(), recurrent_cls=self.recurrent_cls.detach())


@dataclass(frozen=True)
class CanViTOutput:
    state: RecurrentState
    glimpse_patches: Tensor  # [B, g*g, backbone_dim]: backbone patch tokens after the last block
    vpe: Tensor | None  # [B, backbone_dim]: the VPE token after the last block


def canvas_attention_schedule(
    *, num_blocks: int, rw_stride: int, enable_reads: bool,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Indices of the backbone blocks followed by a Canvas Attention Read, and by a Write.

    Every rw_stride blocks comes one operation, alternating Read and Write and
    starting with a Read; the last block is always followed by a Write. Disabling
    reads leaves the Write positions unchanged.
    """
    positions = list(range(rw_stride - 1, num_blocks, rw_stride))
    read_after_blocks, write_after_blocks = positions[0::2], positions[1::2]
    if not write_after_blocks or write_after_blocks[-1] != num_blocks - 1:
        write_after_blocks.append(num_blocks - 1)
    return (tuple(read_after_blocks) if enable_reads else ()), tuple(write_after_blocks)


class CanViT(nn.Module):
    def __init__(self, config: CanViTConfig) -> None:
        super().__init__()
        self.config = config
        spec = config.backbone_spec
        self.backbone = ViTBackbone(spec)
        self.read_after_blocks, self.write_after_blocks = canvas_attention_schedule(
            num_blocks=spec.num_blocks, rw_stride=config.rw_stride, enable_reads=config.enable_reads,
        )
        Read, Write = CANVAS_ATTENTION_CLASSES[config.canvas_projections]
        dims = dict(backbone_dim=spec.embed_dim, canvas_dim=config.canvas_dim, num_heads=config.canvas_num_heads)
        self.canvas_reads = nn.ModuleList([Read(**dims) for _ in self.read_after_blocks])
        self.canvas_writes = nn.ModuleList([Write(**dims) for _ in self.write_after_blocks])

        canvas_scale = 1.0 / math.sqrt(config.canvas_dim)
        self.init_canvas_registers = nn.Parameter(
            torch.randn(1, config.num_canvas_registers, config.canvas_dim) * canvas_scale
        )
        self.init_canvas_patch = nn.Parameter(torch.randn(1, 1, config.canvas_dim) * canvas_scale)
        self.init_recurrent_cls = nn.Parameter(torch.randn(1, 1, spec.embed_dim) * (1.0 / math.sqrt(spec.embed_dim)))
        self.backbone_registers = nn.Parameter(torch.empty(1, config.num_backbone_registers, spec.embed_dim))
        nn.init.normal_(self.backbone_registers, std=0.02)
        self.vpe = ViewpointEncoding(spec.embed_dim) if config.enable_vpe else None

    @property
    def backbone_dim(self) -> int:
        return self.backbone.spec.embed_dim

    @property
    def canvas_dim(self) -> int:
        return self.config.canvas_dim

    @property
    def patch_size(self) -> int:
        return self.backbone.spec.patch_size

    def init_state(self, *, batch_size: int, canvas_grid_size: int) -> RecurrentState:
        """The state before any glimpse, with a canvas_grid_size×canvas_grid_size canvas."""
        registers = self.init_canvas_registers.expand(batch_size, -1, -1)
        patches = self.init_canvas_patch.expand(batch_size, canvas_grid_size**2, -1)
        return RecurrentState(
            canvas=torch.cat([registers, patches], dim=1),
            recurrent_cls=self.init_recurrent_cls.expand(batch_size, -1, -1),
        )

    def canvas_patches(self, canvas: Tensor) -> Tensor:
        """[B, G*G, canvas_dim], registers dropped."""
        return canvas[:, self.config.num_canvas_registers :]

    def canvas_patch_grid(self, canvas: Tensor) -> Tensor:
        """[B, G, G, canvas_dim]: the canvas patches as a spatial map."""
        patches = self.canvas_patches(canvas)
        grid = math.isqrt(patches.shape[1])
        assert grid * grid == patches.shape[1], f"{patches.shape[1]} canvas patches do not form a square grid"
        return patches.view(patches.shape[0], grid, grid, -1)

    def forward(self, *, glimpse: Tensor, state: RecurrentState, viewpoint: Viewpoint) -> CanViTOutput:
        """Process one glimpse [B, 3, px, px] taken at viewpoint; return the next state."""
        B, _, height, width = glimpse.shape
        assert height == width and height % self.patch_size == 0, f"glimpse must be square in whole patches: {glimpse.shape}"
        glimpse_grid = height // self.patch_size

        patches = self.backbone.patch_embed(glimpse)
        prefix = [state.recurrent_cls, self.backbone_registers.expand(B, -1, -1)]
        if self.vpe is not None:
            prefix.insert(0, self.vpe(viewpoint).unsqueeze(1).to(patches.dtype))
        n_prefix = sum(t.shape[1] for t in prefix)
        tokens = torch.cat([*prefix, patches], dim=1)

        spec = self.backbone.spec
        backbone_periods = make_rope_periods(head_dim=spec.head_dim, base=spec.rope_base, device=glimpse.device)
        canvas_periods = make_rope_periods(head_dim=self.config.canvas_head_dim, base=spec.rope_base, device=glimpse.device)
        glimpse_positions = viewpoint_grid_coords(viewpoint, size=glimpse_grid).flatten(1, 2)
        glimpse_rope = compute_2d_rope(positions=glimpse_positions, periods=backbone_periods)
        glimpse_rope_canvas = compute_2d_rope(positions=glimpse_positions, periods=canvas_periods)

        canvas = state.canvas
        canvas_grid = math.isqrt(canvas.shape[1] - self.config.num_canvas_registers)
        canvas_positions = grid_coords(size=canvas_grid, device=canvas.device).flatten(0, 1).unsqueeze(0).expand(B, -1, -1)
        canvas_rope = compute_2d_rope(positions=canvas_positions, periods=canvas_periods)

        read_after_block = dict(zip(self.read_after_blocks, self.canvas_reads))
        write_after_block = dict(zip(self.write_after_blocks, self.canvas_writes))
        for i, block in enumerate(self.backbone.blocks):
            tokens = block(tokens, glimpse_rope)
            if (read := read_after_block.get(i)) is not None:
                tokens = tokens + read(query=tokens, kv=canvas, query_rope=glimpse_rope_canvas, kv_rope=canvas_rope)
            if (write := write_after_block.get(i)) is not None:
                canvas = canvas + write(query=canvas, kv=tokens, query_rope=canvas_rope, kv_rope=glimpse_rope_canvas)

        cls_index = 1 if self.vpe is not None else 0
        return CanViTOutput(
            state=RecurrentState(canvas=canvas, recurrent_cls=tokens[:, cls_index : cls_index + 1].contiguous()),
            glimpse_patches=tokens[:, n_prefix:].contiguous(),
            vpe=tokens[:, 0] if self.vpe is not None else None,
        )
