"""flops.json: the Canvas Attention Read/Write cost, asymmetric and with canvas-side projections (QKVO),
at the 32×32 pretraining canvas and the 64×64 inference canvas."""

from dataclasses import replace

from canvit_pytorch import CanViTConfig
from canvit_pytorch.flops import num_glimpse_tokens
from canvit_pytorch.hub.repos import RELEASED_CANVAS_GRID_SIZE, RELEASED_GLIMPSE_SIZE_PX

from canvit_paper_exporter.core import Dataset
from canvit_paper_exporter.flops.canvit import CANVIT_B, read_write_pair_flops


def _read_write_numbers(config: CanViTConfig, canvas_grid_size: int) -> dict:
    def pair(projections: str) -> int:
        return read_write_pair_flops(
            replace(config, canvas_projections=projections),
            glimpse_size_px=RELEASED_GLIMPSE_SIZE_PX, canvas_grid_size=canvas_grid_size,
        )

    asymmetric, qkvo = pair("asymmetric"), pair("qkvo")
    return {
        "canvas_grid": canvas_grid_size,
        "asym_rw_pair_gf": round(asymmetric / 1e9, 3),
        "full_rw_pair_gf": round(qkvo / 1e9, 3),
        "qkvo_rw_pair_ratio": round(qkvo / asymmetric, 1),
    }


def compute() -> dict:
    patch_size = CANVIT_B.backbone_spec.patch_size
    return {
        "canvit": {
            "canvas_dim": CANVIT_B.canvas_dim,
            "n_local": num_glimpse_tokens(CANVIT_B, glimpse_size_px=RELEASED_GLIMPSE_SIZE_PX),
            "n_patches": (RELEASED_GLIMPSE_SIZE_PX // patch_size) ** 2,
            "backbone_regs": CANVIT_B.num_backbone_registers,
            **_read_write_numbers(CANVIT_B, RELEASED_CANVAS_GRID_SIZE),
            "c64": _read_write_numbers(CANVIT_B, 64),
        },
    }


dataset = Dataset(name="flops", compute=compute)
