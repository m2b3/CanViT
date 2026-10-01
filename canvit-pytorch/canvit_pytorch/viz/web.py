"""Write a recorded rollout as a web bundle: PNG layers and manifest.json (docs/viz.md)."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from PIL import Image

from canvit_pytorch.benchmarks.ade20k import IGNORE_LABEL
from canvit_pytorch.policies.entropy import predictive_entropy
from canvit_pytorch.viewpoint import Viewpoint, crop_box_px
from canvit_pytorch.viz.pca import ColorLimits, color_limits, fit_pca, layernorm, project, to_rgb
from canvit_pytorch.viz.record import Rollout

SCHEMA = "canvit-web-bundle-70766501-fb7f-4856-8a88-bb16253c34c5"
PCAProtocol = Literal["paper", "fixed-limits"]


@dataclass(frozen=True)
class Box:
    """A glimpse's square as fractions of the scene's side, from the top-left corner."""

    top: float
    left: float
    size: float


def glimpse_box(viewpoint: Viewpoint) -> Box:
    top, left, bottom, _ = crop_box_px(viewpoint, image_size_px=1)[0].tolist()
    return Box(top=top, left=left, size=bottom - top)


def entropy_fraction(logits: NDArray[np.floating]) -> NDArray[np.float64]:
    """[C, H, W] logits -> each cell's predictive entropy divided by its maximum, log C."""
    entropy = predictive_entropy(torch.from_numpy(np.asarray(logits, dtype=np.float64))[None])[0]
    return entropy.numpy() / math.log(logits.shape[0])


def canvas_change(before: NDArray[np.floating], after: NDArray[np.floating]) -> NDArray[np.float64]:
    """[N, D] canvas patches before and after a glimpse -> [N] one minus the cosine similarity of the
    layer-normalized patches, in [0, 2]."""
    a, b = layernorm(before), layernorm(after)
    return 1.0 - (a * b).sum(axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


def pixel_accuracy(logits: NDArray[np.floating], annotation: NDArray[np.integer]) -> float:
    """Fraction of the annotated pixels whose class, decoded at the annotation's resolution, is right.

    The [C, G, G] logits are upsampled bilinearly to the [H, W] annotation, as the paper's ADE20K evaluation does.
    """
    upsampled = F.interpolate(torch.from_numpy(np.asarray(logits))[None], size=annotation.shape, mode="bilinear",
                              align_corners=False)
    predicted = upsampled[0].argmax(dim=0).numpy()
    labeled = annotation != IGNORE_LABEL
    return float((predicted[labeled] == annotation[labeled]).mean())


def cell_center_labels(annotation: NDArray[np.integer], grid: int) -> NDArray[np.uint8]:
    """The annotation at the center pixel of each cell of a grid × grid tiling, for display beside the canvas."""
    height, width = annotation.shape
    assert height == width and height % grid == 0, (annotation.shape, grid)
    cell = height // grid
    return annotation[cell // 2 :: cell, cell // 2 :: cell].astype(np.uint8)


def _gray16(fraction: NDArray[np.floating]) -> NDArray[np.uint16]:
    """Fractions in [0, 1] as 16-bit samples: small values keep their detail."""
    return (np.clip(fraction, 0.0, 1.0) * 65535).round().astype(np.uint16)


def _save(array: NDArray[np.uint8] | NDArray[np.uint16], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, optimize=True)


def write_bundle(
    rollout: Rollout,
    out_dir: Path,
    *,
    title: str,
    scene: dict[str, Any],
    model: dict[str, Any],
    readout: dict[str, Any],
    policy: dict[str, Any],
    pca_protocol: PCAProtocol,
    provenance: dict[str, Any],
) -> Path:
    """Write scene.png, each glimpse's PNG layers and manifest.json; return the manifest's path.

    The dictionaries go into the manifest as they are (docs/viz.md lists their fields).
    """
    grid = rollout.canvas_grid_size
    _save(rollout.scene, out_dir / "scene.png")
    if rollout.annotation is not None:
        assert rollout.annotation.shape == rollout.scene.shape[:2], (rollout.annotation.shape, rollout.scene.shape)
        _save(cell_center_labels(rollout.annotation, grid), out_dir / "truth.png")

    basis = fit_pca(rollout.glimpses[-1].canvas)
    projections = [project(basis, g.canvas) for g in rollout.glimpses]
    shared = color_limits(*projections)

    def limits(i: int) -> ColorLimits:
        return shared if pca_protocol == "fixed-limits" else color_limits(projections[i])

    # The canvas before the first glimpse, every patch the same learned vector, in the shared basis and limits.
    _save(to_rgb(project(basis, rollout.initial_canvas), shared).reshape(grid, grid, 3), out_dir / "initial_canvas.png")

    entries = []
    previous = rollout.initial_canvas
    for i, glimpse in enumerate(rollout.glimpses):
        d = f"t{glimpse.t:02d}"
        layers = {name: f"{d}/{name}.png" for name in ("crop", "canvas", "labels", "entropy", "change")}
        _save(glimpse.crop, out_dir / layers["crop"])
        _save(to_rgb(projections[i], limits(i)).reshape(grid, grid, 3), out_dir / layers["canvas"])
        labels = glimpse.logits.argmax(axis=0)
        assert labels.max() < 256
        _save(labels.astype(np.uint8), out_dir / layers["labels"])
        _save(_gray16(entropy_fraction(glimpse.logits)), out_dir / layers["entropy"])
        _save(_gray16(canvas_change(previous, glimpse.canvas) / 2).reshape(grid, grid), out_dir / layers["change"])
        canvas = previous
        for k, residual in enumerate(glimpse.write_residuals):
            # One basis per Write, as in the paper's canvas-evolution figure.
            residual_projection = project(fit_pca(residual), residual)
            layers[f"write{k}"] = f"{d}/write{k}.png"
            rgb = to_rgb(residual_projection, color_limits(residual_projection)).reshape(grid, grid, 3)
            _save(rgb, out_dir / layers[f"write{k}"])
            # The canvas after this Write, in the canvas layer's basis and limits, so a renderer can show it change.
            canvas = canvas + residual
            layers[f"write{k}_canvas"] = f"{d}/write{k}_canvas.png"
            _save(to_rgb(project(basis, canvas), limits(i)).reshape(grid, grid, 3), out_dir / layers[f"write{k}_canvas"])
        previous = glimpse.canvas
        row, col = glimpse.viewpoint.centers[0].tolist()
        entry: dict[str, Any] = {
            "t": glimpse.t,
            "viewpoint": {"row": row, "col": col, "scale": glimpse.viewpoint.scales[0].item()},
            "box": asdict(glimpse_box(glimpse.viewpoint)),
            "layers": layers,
        }
        if rollout.annotation is not None:
            entry["pixel_accuracy"] = pixel_accuracy(glimpse.logits, rollout.annotation)
        entries.append(entry)

    manifest = {
        "schema": SCHEMA,
        "title": title,
        "scene": {"image": "scene.png", "px": int(rollout.scene.shape[0]), **scene}
        | ({"truth": "truth.png"} if rollout.annotation is not None else {}),
        "model": model,
        "readout": readout,
        "policy": policy,
        "canvas_grid": grid,
        "glimpse_px": rollout.glimpse_size_px,
        "pca": {"protocol": pca_protocol, "basis": "last-glimpse"},
        "initial_canvas": "initial_canvas.png",
        "glimpses": entries,
        "provenance": provenance,
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1) + "\n")
    return path
