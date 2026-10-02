"""Generate images for architecture overview diagram.

Shows canvas evolution across multiple timesteps with different viewpoints.

Usage:
    uv run python -m canvit_paper_exporter.run arch_overview
    uv run python -m canvit_paper_exporter.diagrams.arch_overview --n-timesteps 5

Outputs to: diagrams/outputs/arch_overview/
    - scene.png: source image
    - glimpse_t{i}.png: input glimpse at timestep i
    - local_out_t{i}.png: local tokens after backbone at timestep i
    - canvas_out_t{i}.png: canvas state after timestep i (shared PCA from final)
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from canvit_pytorch import Viewpoint, sample_at_viewpoint

from canvit_paper_exporter.diagrams._common import DIAGRAMS_OUTPUTS, BaseConfig, load_canvit
from canvit_paper_exporter.diagrams.io import load_image, save_image, save_pca, save_tensor_rgb

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Viewpoint sequence: (center_y, center_x, scale)
VIEWPOINTS = [
    (-0.7, -0.7, 0.2),
    (0.3, 0.3, 0.5),
    (-0.8, 0.25, 0.15),
]


@dataclass
class Config(BaseConfig):
    n_timesteps: int = 3
    output_dir: Path = DIAGRAMS_OUTPUTS / "arch_overview"


def main(cfg: Config) -> None:
    device = torch.device(cfg.device)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading model: {cfg.model}")
    model = load_canvit(cfg.model, device)
    log.info(f"  canvas_dim={model.canvas_dim}, backbone_dim={model.backbone_dim}")

    log.info(f"Loading image: {cfg.image}")
    image, pil_img = load_image(cfg.image, device)
    save_image(pil_img, cfg.output_dir / "scene.png")
    log.info("  scene.png")

    viewpoints = [
        Viewpoint(
            centers=torch.tensor([[cy, cx]], device=device, dtype=torch.float32),
            scales=torch.tensor([s], device=device, dtype=torch.float32),
        )
        for cy, cx, s in VIEWPOINTS[:cfg.n_timesteps]
    ]

    state = model.init_state(batch_size=1, canvas_grid_size=cfg.canvas_grid)
    canvases: list[Tensor] = []

    log.info(f"Running {cfg.n_timesteps} timesteps...")
    with torch.inference_mode():
        for t, vp in enumerate(viewpoints):
            glimpse = sample_at_viewpoint(spatial=image, viewpoint=vp, glimpse_size_px=cfg.glimpse_px)
            save_tensor_rgb(glimpse[0], cfg.output_dir / f"glimpse_t{t}.png")
            log.info(f"  glimpse_t{t}.png")

            out = model(glimpse=glimpse, state=state, viewpoint=vp)
            state = out.state
            canvases.append(model.canvas_patches(state.canvas)[0].cpu())

            save_pca(out.glimpse_patches[0].cpu(), cfg.output_dir / f"local_out_t{t}.png")
            log.info(f"  local_out_t{t}.png")

    # Shared PCA: fit on final canvas, apply to all
    log.info("Saving canvas states (shared PCA from final)...")
    pca = save_pca(canvases[-1], cfg.output_dir / f"canvas_out_t{len(canvases)-1}.png")
    log.info(f"  canvas_out_t{len(canvases)-1}.png")
    for t in range(len(canvases) - 1):
        save_pca(canvases[t], cfg.output_dir / f"canvas_out_t{t}.png", pca)
        log.info(f"  canvas_out_t{t}.png")

    metadata = {
        "n_timesteps": cfg.n_timesteps,
        "viewpoints": [
            {"x": cx, "y": cy, "scale": s}
            for cy, cx, s in VIEWPOINTS[:cfg.n_timesteps]
        ],
    }
    meta_path = cfg.output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    log.info("  metadata.json")

    log.info(f"Done: {cfg.output_dir}")


if __name__ == "__main__":
    import tyro
    main(tyro.cli(Config))
