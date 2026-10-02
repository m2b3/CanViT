"""Canvit Overview: generate per-glimpse PNGs consumed by canvit_overview_content.typ.

Usage:
    uv run python -m canvit_paper_exporter.run canvit_overview         # py prep + typst compose
    uv run python -m canvit_paper_exporter.diagrams.canvit_overview    # py prep only (PNG bundle)
"""

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from PIL import Image

from canvit_pytorch import CanViT, Viewpoint, sample_at_viewpoint
from canvit_pytorch.preprocess import preprocess as make_preprocess
from canvit_pytorch.viz.pca import fit_pca

from canvit_paper_exporter.diagrams._common import DIAGRAMS_INPUTS, DIAGRAMS_OUTPUTS, BaseConfig, load_canvit
from canvit_paper_exporter.diagrams.io import denormalized_numpy, pca_colors

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Viewpoints: (scale, center_y, center_x). Valid range: center +/- scale in [-1, 1].
# Provide n_images x n_vps_per_image entries; they are split evenly across images.
_DEFAULT_IMAGES = ["Places365_IMG_9600.jpeg"]
_DEFAULT_VIEWPOINTS: list[tuple[float, float, float]] = [
    (0.50, -0.4, -0.4),
    (0.40, -0.18,  0.6),
    (0.30,  0.6, -0.6),
    (0.20,  0.7,  0.5),
]

VP_COLORS = ["white"]


def _draw_grid(ax: Axes, nh: int, nw: int, *, step: float, lw: float, alpha: float) -> None:
    """Draw interior grid lines. The -0.5 offset aligns with imshow's pixel-center convention."""
    for i in range(1, nh):
        ax.axhline(i * step - 0.5, color="white", linewidth=lw, alpha=alpha)
    for i in range(1, nw):
        ax.axvline(i * step - 0.5, color="white", linewidth=lw, alpha=alpha)


def _vp_to_token_rect(center: tuple[float, float], scale: float,
                      grid: int) -> tuple[float, float, float]:
    """VP in normalized [-1, 1]^2 coords -> (row, col, size) in token-index space.

    A VP with center (cy, cx) and scale s covers [cy-s, cy+s] x [cx-s, cx+s].
    Token index i spans normalized range [2i/grid - 1, 2(i+1)/grid - 1].
    """
    cy, cx = center
    r = (cy - scale + 1) / 2 * grid
    c = (cx - scale + 1) / 2 * grid
    return r, c, scale * grid


def vp_rect(ax: Axes, center: tuple[float, float], scale: float,
            grid: int, color: str, *, lw: float, alpha: float) -> None:
    r, c, sz = _vp_to_token_rect(center, scale, grid)
    ax.add_patch(Rectangle(
        (c - 0.5, r - 0.5), sz, sz,
        linewidth=lw, edgecolor=color, facecolor="none", linestyle="--", alpha=alpha))


def make_viewpoint(center: tuple[float, float], scale: float,
                   device: torch.device) -> Viewpoint:
    cy, cx = center
    assert cy - scale >= -1 and cy + scale <= 1, (
        f"VP out of bounds: y={cy}+/-{scale}, valid y in [{-1+scale}, {1-scale}]")
    assert cx - scale >= -1 and cx + scale <= 1, (
        f"VP out of bounds: x={cx}+/-{scale}, valid x in [{-1+scale}, {1-scale}]")
    return Viewpoint(
        centers=torch.tensor([[cy, cx]], dtype=torch.float32, device=device),
        scales=torch.tensor([scale], dtype=torch.float32, device=device),
    )


def _canvas_crop_at_vp(canvas: np.ndarray, center: tuple[float, float],
                       scale: float, grid: int) -> tuple[np.ndarray, int, int, tuple[int, int]]:
    """Tightest bounding box of canvas tokens encapsulating the glimpse.

    Returns (tokens [h*w, D], h, w, (r0, c0) origin in full grid).
    """
    cy, cx = center
    r0 = max(0, int(np.floor((cy - scale + 1) / 2 * grid)))
    r1 = min(grid, int(np.ceil((cy + scale + 1) / 2 * grid)))
    c0 = max(0, int(np.floor((cx - scale + 1) / 2 * grid)))
    c1 = min(grid, int(np.ceil((cx + scale + 1) / 2 * grid)))
    crop = canvas.reshape(grid, grid, -1)[r0:r1, c0:c1]
    return crop.reshape(-1, crop.shape[-1]), r1 - r0, c1 - c0, (r0, c0)


def _cosine_dissimilarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = (a * b).sum(axis=-1)
    norm = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8
    return 1.0 - dot / norm


class GlimpseRow(NamedTuple):
    glimpse_rgb: np.ndarray
    local: np.ndarray
    canvas: np.ndarray
    change: np.ndarray
    label: str


def _extract_row(model: CanViT, out, glimpse: torch.Tensor, prev_canvas: np.ndarray):
    local = out.glimpse_patches[0].detach().numpy()
    canvas = model.canvas_patches(out.state.canvas)[0].detach().numpy()
    glimpse_rgb = denormalized_numpy(glimpse[0])
    change = _cosine_dissimilarity(prev_canvas, canvas)
    return glimpse_rgb, local, canvas, change


def run_example(model: CanViT, ex: dict, *, canvas_grid: int,
                glimpse_px: int, inputs_dir: Path, device: torch.device,
                transform):
    image = transform(Image.open(inputs_dir / ex["image"]).convert("RGB"))
    assert isinstance(image, torch.Tensor)
    image = image.unsqueeze(0).to(device)
    viewpoints = [make_viewpoint(c, s, device) for c, s in ex["viewpoints"]]

    rows: list[GlimpseRow] = []
    with torch.inference_mode():
        state = model.init_state(batch_size=1, canvas_grid_size=canvas_grid)
        prev_canvas = model.canvas_patches(state.canvas)[0].detach().numpy()
        for t, vp in enumerate(viewpoints):
            glimpse = sample_at_viewpoint(spatial=image, viewpoint=vp, glimpse_size_px=glimpse_px)
            out = model(glimpse=glimpse, state=state, viewpoint=vp)
            state = out.state
            g, loc, c, ch = _extract_row(model, out, glimpse, prev_canvas)
            rows.append(GlimpseRow(g, loc, c, ch, label=rf"$t = {t}$"))
            prev_canvas = c

    return rows


def _save_with_vp_contours(path: Path, img, canvas_grid: int,
                           viewpoints: list[tuple[tuple[float, float], float]], **imshow_kw) -> None:
    def _draw(ax: Axes) -> None:
        ax.imshow(img, interpolation="nearest", **imshow_kw)
        for vi, (center, scale) in enumerate(viewpoints):
            current = vi == len(viewpoints) - 1
            kw = dict(lw=1.0, alpha=1.0) if current else dict(lw=0.5, alpha=0.3)
            vp_rect(ax, center, scale, canvas_grid, VP_COLORS[vi % len(VP_COLORS)], **kw)
    _save_ax(path, _draw)


def _save_ax(path: Path, draw_fn) -> None:
    fig, ax = plt.subplots(figsize=(2, 2))
    draw_fn(ax)
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", pad_inches=0, dpi=600, metadata={"Date": None})
    plt.close(fig)


def _save_row_images(out_dir: Path, glimpse_idx: int,
                     glimpse_rgb: np.ndarray, local: np.ndarray,
                     canvas: np.ndarray, change: np.ndarray,
                     canvas_pca, vp_center: tuple[float, float], vp_scale: float,
                     canvas_grid: int, local_grid: int, patch_px: int,
                     viewpoints: list[tuple[tuple[float, float], float]]) -> None:
    cy, cx = vp_center

    def _draw_glimpse(ax: Axes) -> None:
        ax.imshow(glimpse_rgb)
        _draw_grid(ax, local_grid, local_grid, step=float(patch_px), lw=0.5, alpha=0.8)
    _save_ax(out_dir / f"g{glimpse_idx}_glimpse_s{vp_scale:.2f}_c{cy:.1f}_{cx:.1f}.png", _draw_glimpse)

    crop, crop_h, crop_w, crop_origin = _canvas_crop_at_vp(canvas, vp_center, vp_scale, canvas_grid)
    def _draw_canvcrop(ax: Axes) -> None:
        ax.imshow(pca_colors(canvas_pca, crop, shape=(crop_h, crop_w)), interpolation="nearest")
        _draw_grid(ax, crop_h, crop_w, step=1.0, lw=0.3 * 16.0 / max(crop_h, crop_w), alpha=0.4)
        vp_r, vp_c, vp_sz = _vp_to_token_rect(vp_center, vp_scale, canvas_grid)
        ax.set_xlim(vp_c - crop_origin[1] - 0.5, vp_c - crop_origin[1] + vp_sz - 0.5)
        ax.set_ylim(vp_r - crop_origin[0] + vp_sz - 0.5, vp_r - crop_origin[0] - 0.5)
    _save_ax(out_dir / f"g{glimpse_idx}_canvcrop.png", _draw_canvcrop)

    _save_with_vp_contours(out_dir / f"g{glimpse_idx}_canvas.png",
                           pca_colors(canvas_pca, canvas), canvas_grid, viewpoints)
    _save_with_vp_contours(out_dir / f"g{glimpse_idx}_change.png",
                           change.reshape(canvas_grid, canvas_grid), canvas_grid, viewpoints,
                           cmap="magma")


@dataclass
class Config(BaseConfig):
    output: str = "canvit_overview"
    canvas_grid: int = 64
    inputs_dir: Path = DIAGRAMS_INPUTS
    images: list[str] = field(default_factory=lambda: list(_DEFAULT_IMAGES))
    viewpoints: list[tuple[float, float, float]] = field(
        default_factory=lambda: list(_DEFAULT_VIEWPOINTS)
    )


def main(cfg: Config) -> None:
    device = torch.device(cfg.device)

    assert len(cfg.viewpoints) % len(cfg.images) == 0, (
        f"len(viewpoints)={len(cfg.viewpoints)} must be divisible by len(images)={len(cfg.images)}")
    n_vps = len(cfg.viewpoints) // len(cfg.images)
    examples = [
        {
            "image": img,
            "viewpoints": [((cy, cx), s) for s, cy, cx in cfg.viewpoints[i * n_vps:(i + 1) * n_vps]],
        }
        for i, img in enumerate(cfg.images)
    ]

    log.info("Loading model: %s", cfg.model)
    model = load_canvit(cfg.model, device)
    transform = make_preprocess(cfg.canvas_grid * model.patch_size)

    log.info("Running %d examples...", len(examples))
    all_examples = [
        (run_example(model, ex, canvas_grid=cfg.canvas_grid,
                     glimpse_px=cfg.glimpse_px, inputs_dir=cfg.inputs_dir,
                     device=device, transform=transform), ex)
        for ex in examples
    ]

    DIAGRAMS_OUTPUTS.mkdir(parents=True, exist_ok=True)
    base_dir = DIAGRAMS_OUTPUTS / cfg.output
    base_dir.mkdir(parents=True, exist_ok=True)

    log.info("Saving individual images...")
    for rows, ex in all_examples:
        img_stem = Path(ex["image"]).stem
        img_dir = base_dir / img_stem
        # Clean before writing: filenames encode viewpoint coords, so stale
        # files from a previous run with different viewpoints would persist.
        if img_dir.exists():
            shutil.rmtree(img_dir)
        img_dir.mkdir(parents=True, exist_ok=True)
        canvas_pca = fit_pca(rows[-1].canvas)
        for ri, row in enumerate(rows):
            vp_center, vp_scale = ex["viewpoints"][ri]
            local_grid = int(np.sqrt(row.local.shape[0]))
            patch_px = cfg.glimpse_px // local_grid
            _save_row_images(img_dir, ri, row.glimpse_rgb, row.local, row.canvas, row.change,
                             canvas_pca, vp_center, vp_scale, cfg.canvas_grid,
                             local_grid, patch_px, viewpoints=ex["viewpoints"][:ri + 1])
            log.info("  %s/g%d_*", img_stem, ri)

        coords = [
            f"s{s:.2f}_c{cy:.1f}_{cx:.1f}"
            for (cy, cx), s in ex["viewpoints"]
        ]
        meta = {
            "image": ex["image"],
            "dir_name": img_stem,
            "n_glimpses": len(ex["viewpoints"]),
            "glimpse_coords": coords,
            "canvas_grid": cfg.canvas_grid,
            "glimpse_px": cfg.glimpse_px,
        }
        (img_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        log.info("  %s/metadata.json", img_stem)


if __name__ == "__main__":
    import tyro
    main(tyro.cli(Config))
