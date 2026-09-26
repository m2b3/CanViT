"""Record CanViT along a smooth viewpoint path, with its canvas carried and reset at every viewpoint, as a web bundle.

Every sample is one glimpse. With the canvas carried, each glimpse updates the
state left by all earlier ones; reset, each glimpse starts from the initial
state, so its readouts show what one glimpse alone gives. Per sample and
condition the bundle holds the canvas (PCA to RGB), the segmentation, its
entropy, and, when the scene is annotated, the fraction of annotated pixels
labeled correctly. Layers are tiled into atlases (one tile per sample).
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor

from canvit_pytorch.model.canvit import RecurrentState
from canvit_pytorch.model.segmentation import CanViTForSemanticSegmentation
from canvit_pytorch.policies.entropy import predictive_entropy
from canvit_pytorch.preprocess import imagenet_denormalize
from canvit_pytorch.viewpoint import Viewpoint, sample_at_viewpoint
from canvit_pytorch.viz.path import Segment
from canvit_pytorch.viz.pca import ColorLimits, PCABasis, fit_pca, project, to_rgb
from canvit_pytorch.viz.web import cell_center_labels, pixel_accuracy

SCHEMA = "canvit-path-bundle-df96391b-61d4-49cd-a1e5-5a4af9a42e77"
RESET_BATCH_SIZE = 20


@dataclass
class ConditionLayers:
    canvas: list[NDArray[np.uint8]]  # [G, G, 3] per sample
    labels: list[NDArray[np.uint8]]  # [G, G]
    entropy: list[NDArray[np.uint8]]  # [G, G], entropy / log(num_classes) × 255
    pixel_accuracy: list[float]


def _uint8_image(normalized: Tensor) -> NDArray[np.uint8]:
    return (imagenet_denormalize(normalized.float().cpu()).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def _reference_colors(
    model: CanViTForSemanticSegmentation, image: Tensor, *, canvas_grid_size: int, glimpse_size_px: int,
) -> tuple[PCABasis, ColorLimits]:
    """Basis and color limits from the canvas after one full-scene glimpse from the initial state; fixed for
    every sample of both conditions, so a color means the same feature direction throughout."""
    viewpoint = Viewpoint.full_scene(batch_size=1, device=image.device)
    state = model.init_state(batch_size=1, canvas_grid_size=canvas_grid_size)
    glimpse = sample_at_viewpoint(spatial=image, viewpoint=viewpoint, glimpse_size_px=glimpse_size_px)
    canvas = model.canvit.canvas_patches(model.canvit(glimpse=glimpse, state=state, viewpoint=viewpoint).state.canvas)
    tokens = canvas[0].float().cpu().numpy()
    basis = fit_pca(tokens)
    projection = project(basis, tokens)
    return basis, ColorLimits(low=np.percentile(projection, 1, axis=0), high=np.percentile(projection, 99, axis=0))


def _add(layers: ConditionLayers, model: CanViTForSemanticSegmentation, canvas: Tensor, *, basis: PCABasis,
         limits: ColorLimits, annotation: NDArray[np.int64] | None) -> None:
    """Append the display layers of each canvas of a [B, tokens, canvas_dim] batch."""
    grid = math.isqrt(canvas.shape[1] - model.canvit.config.num_canvas_registers)
    logits = model.logits(canvas)
    entropy = (predictive_entropy(logits) / math.log(logits.shape[1])).cpu().numpy()
    patches = model.canvit.canvas_patches(canvas).float().cpu().numpy()
    for i in range(canvas.shape[0]):
        layers.canvas.append(to_rgb(project(basis, patches[i]), limits).reshape(grid, grid, 3))
        layers.labels.append(logits[i].argmax(dim=0).cpu().numpy().astype(np.uint8))
        layers.entropy.append((np.clip(entropy[i], 0, 1) * 255).round().astype(np.uint8))
        if annotation is not None:
            layers.pixel_accuracy.append(pixel_accuracy(logits[i].cpu().numpy(), annotation))


@torch.inference_mode()
def record_path(
    model: CanViTForSemanticSegmentation, image: Tensor, viewpoints: list[Viewpoint], *,
    canvas_grid_size: int, glimpse_size_px: int, annotation: NDArray[np.int64] | None,
) -> tuple[dict[str, ConditionLayers], list[NDArray[np.uint8]]]:
    """Layers per condition ("carried", "reset") and the model's input crops, for a [1, 3, H, W] image."""
    assert image.shape[0] == 1, image.shape
    basis, limits = _reference_colors(model, image, canvas_grid_size=canvas_grid_size, glimpse_size_px=glimpse_size_px)
    carried, reset = ConditionLayers([], [], [], []), ConditionLayers([], [], [], [])
    crops: list[NDArray[np.uint8]] = []

    state = model.init_state(batch_size=1, canvas_grid_size=canvas_grid_size)
    for viewpoint in viewpoints:
        viewpoint = Viewpoint(centers=viewpoint.centers.to(image.device), scales=viewpoint.scales.to(image.device))
        glimpse = sample_at_viewpoint(spatial=image, viewpoint=viewpoint, glimpse_size_px=glimpse_size_px)
        crops.append(_uint8_image(glimpse[0]))
        state = model.canvit(glimpse=glimpse, state=state, viewpoint=viewpoint).state
        _add(carried, model, state.canvas, basis=basis, limits=limits, annotation=annotation)

    for start in range(0, len(viewpoints), RESET_BATCH_SIZE):
        chunk = viewpoints[start : start + RESET_BATCH_SIZE]
        viewpoint = Viewpoint(centers=torch.cat([v.centers for v in chunk]).to(image.device),
                              scales=torch.cat([v.scales for v in chunk]).to(image.device))
        images = image.expand(len(chunk), -1, -1, -1)
        glimpses = sample_at_viewpoint(spatial=images, viewpoint=viewpoint, glimpse_size_px=glimpse_size_px)
        initial: RecurrentState = model.init_state(batch_size=len(chunk), canvas_grid_size=canvas_grid_size)
        canvas = model.canvit(glimpse=glimpses, state=initial, viewpoint=viewpoint).state.canvas
        _add(reset, model, canvas, basis=basis, limits=limits, annotation=annotation)
    return {"carried": carried, "reset": reset}, crops


def _atlas(tiles: list[NDArray[np.uint8]], columns: int) -> NDArray[np.uint8]:
    rows = math.ceil(len(tiles) / columns)
    height, width = tiles[0].shape[:2]
    atlas = np.zeros((rows * height, columns * width, *tiles[0].shape[2:]), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, columns)
        atlas[r * height : (r + 1) * height, c * width : (c + 1) * width] = tile
    return atlas


def write_path_bundle(
    out_dir: Path, *, title: str, scene: NDArray[np.uint8], scene_meta: dict[str, Any], annotation: NDArray[np.int64] | None,
    keypoints: list[tuple[float, float, float]], segments: list[Segment], viewpoints: list[Viewpoint], duration_ms: int,
    conditions: dict[str, ConditionLayers], crops: list[NDArray[np.uint8]], canvas_grid_size: int,
    model: dict[str, Any], readout: dict[str, Any], provenance: dict[str, Any],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = math.ceil(math.sqrt(len(viewpoints)))
    Image.fromarray(scene).save(out_dir / "scene.png", optimize=True)
    Image.fromarray(_atlas(crops, columns)).save(out_dir / "inputs.jpg", quality=90)
    if annotation is not None:
        Image.fromarray(cell_center_labels(annotation, canvas_grid_size)).save(out_dir / "truth.png", optimize=True)
    condition_entries = {}
    for name, layers in conditions.items():
        files = {}
        for layer in ("canvas", "labels", "entropy"):
            files[layer] = f"{name}-{layer}.png"
            Image.fromarray(_atlas(getattr(layers, layer), columns)).save(out_dir / files[layer], optimize=True)
        condition_entries[name] = {"layers": files} | (
            {"pixel_accuracy": layers.pixel_accuracy} if annotation is not None else {}
        )
    manifest = {
        "schema": SCHEMA,
        "title": title,
        "scene": {"image": "scene.png", "px": int(scene.shape[0]), **scene_meta}
        | ({"truth": "truth.png"} if annotation is not None else {}),
        "model": model,
        "readout": readout,
        "path": {"keypoints": keypoints, "segments": segments, "duration_ms": duration_ms,
                 "viewpoints": [[*v.centers[0].tolist(), v.scales[0].item()] for v in viewpoints]},
        "canvas_grid": canvas_grid_size,
        "glimpse_px": crops[0].shape[0],
        "atlas_columns": columns,
        "inputs": "inputs.jpg",
        "pca": {"basis": "canvas after one full-scene glimpse from the initial state",
                "limits": "1st and 99th percentiles of that canvas's projection, fixed for all samples"},
        "conditions": condition_entries,
        "provenance": provenance,
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1) + "\n")
    return path
