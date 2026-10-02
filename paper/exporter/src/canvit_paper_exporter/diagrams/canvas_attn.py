"""Generate images for canvas attention diagram.

Full-scene glimpse at t=0 on a fresh canvas, capturing intermediate
states around a single R/W pair (default: last).

Usage:
    uv run python -m canvit_paper_exporter.run canvas_attention_combined
    uv run python -m canvit_paper_exporter.diagrams.canvas_attn --target-pair 0

Outputs to: diagrams/outputs/canvas_attn/
    - scene.png, glimpse_t0.png: context
    - glimpse_before_read.png, glimpse_after_read.png: local tokens around read
    - glimpse_before_write.png: local tokens entering write
    - canvas_in.png, canvas_after_write.png: canvas before/after write
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from canvit_pytorch import CanViT, Viewpoint, sample_at_viewpoint

from canvit_paper_exporter.diagrams._common import DIAGRAMS_OUTPUTS, BaseConfig, load_canvit
from canvit_paper_exporter.diagrams.io import load_image, save_image, save_pca, save_tensor_rgb

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

FULL_SCENE_VP = (0.0, 0.0, 1.0)


@dataclass
class CapturedStates:
    glimpse_before_read: Tensor
    canvas_in: Tensor
    glimpse_after_read: Tensor
    glimpse_before_write: Tensor
    canvas_after_write: Tensor


def _patches(local: Tensor, model: CanViT) -> Tensor:
    """Extract patch tokens from the glimpse stream: [vpe?, recurrent_cls, registers, PATCHES]."""
    n_regs = model.config.num_backbone_registers
    assert model.backbone_registers.shape == (1, n_regs, model.backbone_dim)
    has_vpe = model.vpe is not None
    n_prefix = (1 if has_vpe else 0) + 1 + n_regs
    return local[:, n_prefix:]


def _spatial(canvas: Tensor, model: CanViT) -> Tensor:
    return model.canvas_patches(canvas)


class IntermediateCapture:
    """Capture intermediate states around a single R/W pair via hooks.

    canvas_read/write return the cross-attention DELTA (residual added outside).
    So: read_post output is the delta, write_post output is the delta.
    We reconstruct the post-residual state from pre + delta.
    """

    def __init__(self, model: CanViT, target_pair: int):
        self.model = model
        self.target_pair = target_pair
        self.states: dict[str, Tensor] = {}
        self.hooks: list[Any] = []

    def __enter__(self) -> "IntermediateCapture":
        read_mod = self.model.canvas_reads[self.target_pair]
        write_mod = self.model.canvas_writes[self.target_pair]

        def read_pre(module: Any, args: tuple, kwargs: dict) -> None:
            self.states["glimpse_before_read"] = _patches(kwargs["query"], self.model).clone()
            self.states["canvas_in"] = _spatial(kwargs["kv"], self.model).clone()

        def read_post(module: Any, args: tuple, kwargs: dict, output: Tensor) -> None:
            self.states["glimpse_after_read"] = (
                _patches(kwargs["query"], self.model) + _patches(output, self.model)
            ).clone()

        def write_pre(module: Any, args: tuple, kwargs: dict) -> None:
            self.states["glimpse_before_write"] = _patches(kwargs["kv"], self.model).clone()

        def write_post(module: Any, args: tuple, kwargs: dict, output: Tensor) -> None:
            self.states["canvas_after_write"] = (
                _spatial(kwargs["query"], self.model) + _spatial(output, self.model)
            ).clone()

        self.hooks.append(read_mod.register_forward_pre_hook(read_pre, with_kwargs=True))
        self.hooks.append(read_mod.register_forward_hook(read_post, with_kwargs=True))
        self.hooks.append(write_mod.register_forward_pre_hook(write_pre, with_kwargs=True))
        self.hooks.append(write_mod.register_forward_hook(write_post, with_kwargs=True))
        return self

    def __exit__(self, *args: Any) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get(self) -> CapturedStates:
        return CapturedStates(**self.states)


@dataclass
class Config(BaseConfig):
    target_pair: int = 2
    output_dir: Path = DIAGRAMS_OUTPUTS / "canvas_attn"


def main(cfg: Config) -> None:
    device = torch.device(cfg.device)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading model: {cfg.model}")
    model = load_canvit(cfg.model, device)

    log.info(f"Loading image: {cfg.image}")
    image, pil_img = load_image(cfg.image, device)
    save_image(pil_img, cfg.output_dir / "scene.png")
    log.info("  scene.png")

    assert 0 <= cfg.target_pair < len(model.read_after_blocks)
    cy, cx, sc = FULL_SCENE_VP
    vp = Viewpoint(
        centers=torch.tensor([[cy, cx]], device=device, dtype=torch.float32),
        scales=torch.tensor([sc], device=device, dtype=torch.float32),
    )

    state = model.init_state(batch_size=1, canvas_grid_size=cfg.canvas_grid)

    with torch.inference_mode():
        log.info(f"t=0: full-scene glimpse, capturing R/W pair {cfg.target_pair}...")
        glimpse = sample_at_viewpoint(spatial=image, viewpoint=vp, glimpse_size_px=cfg.glimpse_px)
        save_tensor_rgb(glimpse[0], cfg.output_dir / "glimpse_t0.png")
        log.info("  glimpse_t0.png")

        with IntermediateCapture(model, target_pair=cfg.target_pair) as cap:
            model(glimpse=glimpse, state=state, viewpoint=vp)
            s = cap.get()

    log.info(f"Captured: glimpse {s.glimpse_before_read.shape}, canvas {s.canvas_in.shape}")

    log.info("Saving glimpse states...")
    g_pca = save_pca(s.glimpse_before_write[0], cfg.output_dir / "glimpse_before_write.png")
    log.info("  glimpse_before_write.png")
    save_pca(s.glimpse_before_read[0], cfg.output_dir / "glimpse_before_read.png", g_pca)
    log.info("  glimpse_before_read.png")
    save_pca(s.glimpse_after_read[0], cfg.output_dir / "glimpse_after_read.png", g_pca)
    log.info("  glimpse_after_read.png")

    log.info("Saving canvas states...")
    c_pca = save_pca(s.canvas_after_write[0], cfg.output_dir / "canvas_after_write.png")
    log.info("  canvas_after_write.png")
    save_pca(s.canvas_in[0], cfg.output_dir / "canvas_in.png", c_pca)
    log.info("  canvas_in.png")

    log.info(f"Done: {cfg.output_dir}")


if __name__ == "__main__":
    import tyro
    main(tyro.cli(Config))
