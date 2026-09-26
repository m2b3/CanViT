"""The per-mask IoU table: one row per (image, class), pixel counts at the label resolution."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from canvit_pytorch.benchmarks import ade20k
from canvit_pytorch.evaluate.tasks.ade20k_segmentation import predicted_labels

log = logging.getLogger(__name__)

RESIZE_MODE = "squish"
"""How benchmarks.ade20k.evaluation_transform resizes scenes: to a square, without cropping."""


def per_image_counts(logits: Tensor, labels: Tensor) -> Tensor:
    """[3, B, num_classes] on the CPU: intersection, union and label area in pixels, predictions upsampled to labels."""
    predictions = predicted_labels(logits, size_px=labels.shape[-1])
    return torch.stack(ade20k.per_image_confusion(predictions, labels, ade20k.NUM_CLASSES)).cpu()


def per_mask_rows(
    *, intersection: Tensor, union: Tensor, label_area: Tensor, mask_resolution_px: int, **run_columns: int,
) -> pd.DataFrame:
    """Rows for every (image, class) of [N, num_classes] pixel counts; run_columns are constant columns.

    Columns: image_idx, class_idx (ADE20K's 1-based class index), *run_columns, inter_px, union_px,
    gt_area_px, mask_resolution_px, resize_mode.
    """
    assert intersection.shape == union.shape == label_area.shape, (intersection.shape, union.shape, label_area.shape)
    num_images, num_classes = intersection.shape
    return pd.DataFrame({
        "image_idx": torch.arange(num_images).repeat_interleave(num_classes).numpy(),
        "class_idx": torch.arange(1, num_classes + 1).repeat(num_images).numpy(),
        **run_columns,
        "inter_px": intersection.flatten().numpy().astype(np.int64),
        "union_px": union.flatten().numpy().astype(np.int64),
        "gt_area_px": label_area.flatten().numpy().astype(np.int64),
        "mask_resolution_px": mask_resolution_px,
        "resize_mode": RESIZE_MODE,
    })


def log_mean_iou(*, intersection: Tensor, union: Tensor, label: str) -> None:
    """Dataset-level mIoU of [N, num_classes] counts, to compare with the ADE20K segmentation task."""
    total_union = union.sum(dim=0)
    present = total_union > 0
    log.info("mIoU %s: %.2f", label, 100 * (intersection.sum(dim=0)[present] / total_union[present]).mean().item())


def check_image_count(num_images: int) -> None:
    """Figure 5 covers the whole validation set; a partial dataset would silently change it."""
    assert num_images == ade20k.NUM_VALIDATION_IMAGES, (
        f"ADE20K validation has {ade20k.NUM_VALIDATION_IMAGES} images, the dataset holds {num_images}"
    )


def write(table: pd.DataFrame, *, output: Path, runs: dict[str, object]) -> None:
    """Write the table to output and a description of the runs that produced it next to it, as JSON."""
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output, index=False)
    output.with_suffix(".json").write_text(json.dumps(runs, indent=2, default=str) + "\n")
    log.info("Saved %s (%d rows)", output, len(table))


def replace_canvas_grid(rows: pd.DataFrame, *, output: Path, canvas_grid_size: int, run: dict[str, object]) -> None:
    """Put one canvas grid's rows into the CanViT table at output, replacing that grid's earlier rows.

    The JSON next to the table describes the run behind each canvas grid, keyed by grid size.
    """
    kept, runs = rows.iloc[:0], {}
    if output.exists():
        existing = pd.read_parquet(output)
        kept = existing.loc[existing["canvas_resolution"] != canvas_grid_size]
        runs = json.loads(output.with_suffix(".json").read_text())
    runs[str(canvas_grid_size)] = run
    grids = {str(grid) for grid in kept["canvas_resolution"].tolist()} | {str(canvas_grid_size)}
    assert grids == set(runs), f"{output}: canvas grids {sorted(grids)}, runs described {sorted(runs)}"
    write(pd.concat([kept, rows], ignore_index=True) if len(kept) else rows, output=output, runs=runs)
