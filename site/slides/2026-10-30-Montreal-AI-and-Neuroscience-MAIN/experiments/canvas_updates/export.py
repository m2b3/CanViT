"""The canvas's updates within a glimpse, for #canvas-updates: the paper's figure "Canvas updates within and across
glimpses" (fig:canvas-evolution) rebuilt for a slide from its generator, paper/exporter's
diagrams/canvas_write_evolution.py. The released CanViT-B takes a candidate's glimpses in turn on a 64 x 64 canvas, as
in the paper; that module's hooks (WriteCapture) capture each Canvas Attention Write's residual and the canvas after
it, and each snapshot is colored by a PCA fitted on itself, as in the paper.

Per candidate, into <out>/<name>/: scene.jpg, g<t>_glimpse.png (the glimpse with its patch grid, as the paper draws
it), g<t>_<stage>.png for each snapshot (the paper's names: initial, write<k>_residual, write<k>_updated; 64 x 64, the
slide upscales them with nearest neighbor), and meta.json (each glimpse's viewpoint and box as fractions of the
scene). The sweep tries every scene of SWEEP_SCENES at every viewpoint of SWEEP_VIEWPOINTS as a first glimpse, and
writes a contact sheet; the deck's candidates go to the deck's data.

    uv run --project ../../../paper/exporter python -m experiments.canvas_updates.export --sweep   # from the deck
"""

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from canvit_paper_exporter.diagrams._common import EXAMPLE_IMAGES, load_canvit
from canvit_paper_exporter.diagrams.canvas_write_evolution import WriteCapture, _save_glimpse_png, _snapshot_filename
from canvit_paper_exporter.diagrams.io import pca_colors, save_photo
from canvit_pytorch import CanViT, sample_at_viewpoint
from canvit_pytorch.benchmarks.ade20k.dataset import dataset_root
from canvit_pytorch.hub.repos import FLAGSHIP, RELEASED_GLIMPSE_SIZE_PX
from canvit_pytorch.preprocess import preprocess
from canvit_pytorch.viz.pca import fit_pca
from matplotlib.patches import Rectangle
from PIL import Image

from experiments.glimpses import ViewpointTuple, viewpoints
from experiments.outputs import DECK_DATA, WORK

log = logging.getLogger(__name__)

CANVAS_GRID = 64  # as the paper's figure


@dataclass(frozen=True)
class Candidate:
    name: str
    source: str  # an example image of canvit-pytorch's test_data (as the paper's figure), or an ADE20K validation id
    viewpoints: tuple[ViewpointTuple, ...]  # (row, col, scale), in turn


def paper(name: str, source: str, viewpoints: list[tuple[float, tuple[float, float]]]) -> Candidate:
    """A scenario of the paper's figure, its (scale, (row, col)) viewpoints as (row, col, scale)."""
    return Candidate(name, source, tuple((row, col, s) for s, (row, col) in viewpoints))


# The paper's two scenarios (canvas_write_evolution.DEFAULT_SCENARIOS), then the deck's own scenes.
PAPER = (
    paper("cat", "Cat03.jpg", [(0.6, (-0.4, -0.4)), (0.6, (-0.3, 0.3)), (0.6, (0.4, 0.4))]),
    paper("food-truck", "Places365_IMG_9600.jpeg", [(0.6, (-0.3, -0.3)), (0.5, (0.4, 0.5)), (0.6, (0.0, 0.0))]),
)
SWEEP_SCENES = {
    "street": "ADE_val_00001780", "living-room": "ADE_val_00001509", "lamp-room": "ADE_val_00001517",
    "seafront-bus": "ADE_val_00000996", "table": "ADE_val_00001271", "bedroom": "ADE_val_00000124",
    "television": "ADE_val_00001715",
}
SWEEP_VIEWPOINTS: tuple[ViewpointTuple, ...] = (
    (0.0, 0.0, 0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5),
)
# The slide's example first (the paper's own cat, its first glimpse), then alternatives chosen by eye from the sweep's
# contact sheet, which the authors can switch to (index.html, canvasUpdatesExample).
DECK: tuple[Candidate, ...] = (
    *PAPER,
    Candidate("bedroom-window", SWEEP_SCENES["bedroom"], ((0.0, 0.0, 0.5),)),
    Candidate("seafront-bus", SWEEP_SCENES["seafront-bus"], ((0.5, -0.5, 0.5),)),
)


@dataclass(frozen=True)
class Config:
    sweep: bool = False
    """every SWEEP_SCENES scene at every SWEEP_VIEWPOINTS viewpoint, as a first glimpse, and the paper's scenarios"""
    work: Path = WORK / "canvas_updates"
    deck: Path = DECK_DATA / "canvas-updates"
    device: str = "mps"


def scene_path(source: str) -> Path:
    if source.startswith("ADE_val_"):
        return dataset_root() / "images/validation" / f"{source}.jpg"
    return EXAMPLE_IMAGES / source


@torch.inference_mode()
def export(candidate: Candidate, model: CanViT, device: torch.device, out: Path) -> None:
    # The scene at the canvas's resolution in pixels, as the paper's generator takes it.
    image = preprocess(CANVAS_GRID * model.patch_size)(Image.open(scene_path(candidate.source)).convert("RGB"))
    image = image[None].to(device)
    state = model.init_state(batch_size=1, canvas_grid_size=CANVAS_GRID)
    glimpses = []
    with WriteCapture(model) as capture:
        for vp in candidate.viewpoints:
            viewpoint = viewpoints([vp], device)
            glimpse = sample_at_viewpoint(spatial=image, viewpoint=viewpoint, glimpse_size_px=RELEASED_GLIMPSE_SIZE_PX)
            glimpses.append(glimpse[0].cpu())
            state = model(glimpse=glimpse, state=state, viewpoint=viewpoint).state
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    save_photo(image[0], out / "scene.jpg", 512)
    for t, glimpse in enumerate(glimpses):
        _save_glimpse_png(glimpse, model.patch_size, out / f"g{t}_glimpse.png")
    for snap in capture.snapshots:
        tokens = model.canvas_patches(snap.data)[0].numpy()
        plt.imsave(out / f"{_snapshot_filename(snap)}.png", pca_colors(fit_pca(tokens), tokens))
    (out / "meta.json").write_text(json.dumps({
        "scene": candidate.source, "model": FLAGSHIP, "canvas_grid": CANVAS_GRID,
        "glimpse_px": RELEASED_GLIMPSE_SIZE_PX, "writes_per_glimpse": len(model.canvas_writes),
        "glimpses": [{"viewpoint": list(vp), "box": {"top": (vp[0] - vp[2] + 1) / 2, "left": (vp[1] - vp[2] + 1) / 2,
                                                     "size": vp[2]}} for vp in candidate.viewpoints],
    }, indent=1))
    log.info("%s: %d glimpses -> %s", candidate.name, len(glimpses), out)


def contact_sheet(outs: list[Path], path: Path, t: int = 0) -> None:
    """Per candidate, glimpse t: the scene with the glimpse's box, the glimpse, each Write's residual and the canvas
    after the last Write, the box drawn on each."""
    stages = ["write0_residual", "write1_residual", "write2_residual", "write2_updated"]
    fig, axes = plt.subplots(len(outs), 2 + len(stages), figsize=(2.0 * (2 + len(stages)), 2.1 * len(outs)),
                             squeeze=False)
    for row, out in zip(axes, outs):
        meta = json.loads((out / "meta.json").read_text())
        box = meta["glimpses"][t]["box"]
        panels = [plt.imread(out / "scene.jpg"), plt.imread(out / f"g{t}_glimpse.png")]
        panels += [plt.imread(out / f"g{t}_{stage}.png") for stage in stages]
        for k, (ax, panel) in enumerate(zip(row, panels)):
            ax.imshow(panel, interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            if k != 1:
                n = panel.shape[0]
                ax.add_patch(Rectangle((box["left"] * n - 0.5, box["top"] * n - 0.5), box["size"] * n, box["size"] * n,
                                       fill=False, edgecolor="white", linewidth=1.2, linestyle="--"))
        row[0].set_ylabel(out.name, fontsize=8)
    for ax, title in zip(axes[0], ["scene", "glimpse", *stages]):
        ax.set_title(title, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90)
    log.info("contact sheet -> %s", path)


def main(cfg: Config) -> None:
    device = torch.device(cfg.device)
    model = load_canvit(FLAGSHIP, device)
    if cfg.sweep:
        sweep = list(PAPER) + [Candidate(f"{name}-{k}", source, (vp,)) for name, source in SWEEP_SCENES.items()
                               for k, vp in enumerate(SWEEP_VIEWPOINTS)]
        outs = []
        for candidate in sweep:
            export(candidate, model, device, cfg.work / "sweep" / candidate.name)
            outs.append(cfg.work / "sweep" / candidate.name)
        contact_sheet(outs, cfg.work / "sweep_contact_sheet.png")
    for candidate in DECK:
        export(candidate, model, device, cfg.deck / candidate.name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(tyro.cli(Config))
