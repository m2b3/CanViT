"""Canvas evolution: PCA colors of the canvas before and after each Canvas Attention Write.

Forward hooks on the Write modules capture each Write's residual and the canvas it produces,
glimpse by glimpse, for the scenes and viewpoints of DEFAULT_SCENARIOS.

Usage:
    uv run python -m canvit_paper_exporter.run canvas_evolution
    uv run python -m canvit_paper_exporter.diagrams.canvas_write_evolution --pca-anchor
"""

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from canvit_pytorch import CanViT, Viewpoint, sample_at_viewpoint
from canvit_pytorch.preprocess import preprocess as make_preprocess
from canvit_pytorch.viz.pca import fit_pca
from PIL import Image
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from canvit_paper_exporter.diagrams._common import DIAGRAMS_INPUTS, DIAGRAMS_OUTPUTS, BaseConfig, load_canvit
from canvit_paper_exporter.diagrams.io import denormalized_numpy, pca_colors

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


@dataclass
class Snapshot:
    label: str
    glimpse_idx: int
    data: Tensor  # [B, N, D]


@dataclass
class WriteCapture:
    """Hooks on every Canvas Attention Write: the canvas entering the first Write of each glimpse,
    then each Write's residual and the canvas after it."""

    model: CanViT
    snapshots: list[Snapshot] = field(default_factory=list)
    _handles: list[RemovableHandle] = field(default_factory=list)
    _glimpse_idx: int = 0
    _writes_this_glimpse: int = 0

    def _snap(self, label: str, data: Tensor) -> None:
        self.snapshots.append(Snapshot(label=label, glimpse_idx=self._glimpse_idx, data=data.detach().cpu()))

    def _after_write(self) -> None:
        self._writes_this_glimpse += 1
        if self._writes_this_glimpse == len(self.model.canvas_writes):
            self._glimpse_idx += 1
            self._writes_this_glimpse = 0

    def __enter__(self) -> "WriteCapture":
        writes = self.model.canvas_writes
        self._handles.append(writes[0].register_forward_pre_hook(
            lambda _mod, _args, kwargs: self._snap("Initial Canvas", kwargs["query"]), with_kwargs=True,
        ))
        for i, write in enumerate(writes):
            def hook(_mod: nn.Module, _args: tuple, kwargs: dict, output: Tensor, label: str = f"Write {i}") -> None:
                self._snap(f"{label}: Residual", output)
                self._snap(f"{label}: Updated Canvas", kwargs["query"] + output)
                self._after_write()
            self._handles.append(write.register_forward_hook(hook, with_kwargs=True))
        return self

    def __exit__(self, *_: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _snapshot_filename(snap: Snapshot) -> str:
    g, label = snap.glimpse_idx, snap.label
    if label == "Initial Canvas":
        stage = "initial"
    else:
        # "Write 0: Residual" -> "write0_residual"; "Write 0: Updated Canvas" -> "write0_updated"
        prefix, kind = label.split(": ", 1)
        stage = f"write{prefix.split()[1]}_{kind.split()[0].lower()}"
    return f"g{g}_{stage}"


def _viewpoint_coords(vp: Viewpoint) -> str:
    s = vp.scales[0].item()
    cy, cx = vp.centers[0, 0].item(), vp.centers[0, 1].item()
    return f"s{s:.2f}_c{cy:.1f}_{cx:.1f}"


def _save_glimpse_png(glimpse: Tensor, patch_px: int, path: Path) -> None:
    """Save denormalized glimpse with patch-grid overlay (semi-transparent white)."""
    img = denormalized_numpy(glimpse)
    alpha = 0.55
    img[patch_px::patch_px, :] = img[patch_px::patch_px, :] * (1 - alpha) + alpha
    img[:, patch_px::patch_px] = img[:, patch_px::patch_px] * (1 - alpha) + alpha
    plt.imsave(path, img)


@dataclass
class Scenario:
    """An image and its viewpoints, each (scale, (center row, center col))."""

    image: str
    viewpoints: list[tuple[float, tuple[float, float]]]


# canvas_evolution_content.typ renders these two scenes side by side. The viewpoints visit distinct
# regions: eyes and background for Cat03; lamp, window and center for Places365.
DEFAULT_SCENARIOS = (
    Scenario("Cat03.jpg",                [(0.6, (-0.4, -0.4)), (0.6, (-0.3,  0.3)), (0.6, (0.4, 0.4))]),
    Scenario("Places365_IMG_9600.jpeg",  [(0.6, (-0.3, -0.3)), (0.5, ( 0.4,  0.5)), (0.6, (0.0, 0.0))]),
)


@dataclass
class Config(BaseConfig):
    output: str = "canvas_evolution"
    canvas_grid: int = 64
    inputs_dir: Path = DIAGRAMS_INPUTS
    pca_anchor: bool = False
    """Color every snapshot of a glimpse in the basis of snapshot pca_anchor_idx instead of its own."""
    pca_anchor_idx: int = -1
    scenarios: tuple[Scenario, ...] = DEFAULT_SCENARIOS


def main(cfg: Config) -> None:
    device = torch.device(cfg.device)
    log.info("Loading model: %s", cfg.model)
    model = load_canvit(cfg.model, device)
    for scenario in cfg.scenarios:
        _generate_one(cfg, scenario, model, device)


@torch.inference_mode()
def _generate_one(cfg: Config, scenario: Scenario, model: CanViT, device: torch.device) -> None:
    transform = make_preprocess(cfg.canvas_grid * model.patch_size)
    image = transform(Image.open(cfg.inputs_dir / scenario.image).convert("RGB"))
    assert isinstance(image, Tensor)
    image = image.unsqueeze(0).to(device)
    log.info("Image: %s  shape=%s  canvas=%dx%d  glimpse=%dpx",
             scenario.image, tuple(image.shape), cfg.canvas_grid, cfg.canvas_grid, cfg.glimpse_px)

    viewpoints = [
        Viewpoint(
            centers=torch.tensor([[cy, cx]], device=device, dtype=torch.float32),
            scales=torch.tensor([s], device=device, dtype=torch.float32),
        )
        for s, (cy, cx) in scenario.viewpoints
    ]
    glimpses: list[Tensor] = []
    state = model.init_state(batch_size=1, canvas_grid_size=cfg.canvas_grid)
    with WriteCapture(model) as capture:
        for vp in viewpoints:
            glimpse = sample_at_viewpoint(spatial=image, viewpoint=vp, glimpse_size_px=cfg.glimpse_px)
            glimpses.append(glimpse[0].cpu())
            state = model(glimpse=glimpse, state=state, viewpoint=vp).state
    snapshots = capture.snapshots
    log.info("Captured %d canvas snapshots", len(snapshots))

    dir_name = re.split(r"[._]", scenario.image, maxsplit=1)[0]  # Cat03.jpg -> Cat03
    run_dir = DIAGRAMS_OUTPUTS / cfg.output / dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("Saving images to %s...", run_dir)

    plt.imsave(run_dir / "scene.png", denormalized_numpy(image[0]))
    log.info("  scene.png")

    by_glimpse: dict[int, list[Snapshot]] = defaultdict(list)
    for snap in snapshots:
        by_glimpse[snap.glimpse_idx].append(snap)
    for g, snaps in sorted(by_glimpse.items()):
        def patches(snap: Snapshot):
            return model.canvas_patches(snap.data)[0].numpy()

        anchor = fit_pca(patches(snaps[cfg.pca_anchor_idx])) if cfg.pca_anchor else None
        for snap in snaps:
            spatial = patches(snap)
            fname = _snapshot_filename(snap) + ".png"
            plt.imsave(run_dir / fname, pca_colors(anchor if anchor is not None else fit_pca(spatial), spatial))
            log.info("  %s", fname)
        gfname = f"g{g}_glimpse_{_viewpoint_coords(viewpoints[g])}.png"
        _save_glimpse_png(glimpses[g], model.patch_size, run_dir / gfname)
        log.info("  %s", gfname)

    metadata = {
        "image": scenario.image,
        "dir_name": dir_name,
        "n_glimpses": len(viewpoints),
        "n_writes": len(model.canvas_writes),
        "glimpse_coords": [_viewpoint_coords(vp) for vp in viewpoints],
        "canvas_grid": cfg.canvas_grid,
        "glimpse_px": cfg.glimpse_px,
        "vp_policy": "manual",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    log.info("  metadata.json")
    log.info("Done: %s", run_dir)


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Config))
