import json
from pathlib import Path

import pandas as pd
import torch

from canvit_pytorch.evaluate.tasks.mask_iou import table


def canvit_rows(canvas_grid_size: int, *, count: int) -> pd.DataFrame:
    counts = torch.full((2, 3), float(count))
    return table.per_mask_rows(
        intersection=counts, union=counts, label_area=counts, mask_resolution_px=512,
        canvas_resolution=canvas_grid_size, timestep=0,
    )


def test_columns_the_paper_pipeline_reads() -> None:
    assert list(canvit_rows(8, count=1).columns) == [
        "image_idx", "class_idx", "canvas_resolution", "timestep", "inter_px", "union_px", "gt_area_px",
        "mask_resolution_px", "resize_mode",
    ]


def test_rerunning_a_canvas_grid_replaces_only_its_rows(tmp_path: Path) -> None:
    output = tmp_path / "canvit_iou.parquet"
    for grid, count in ((8, 1), (16, 2), (8, 3)):
        rows = canvit_rows(grid, count=count)
        table.replace_canvas_grid(rows, output=output, canvas_grid_size=grid, run={"count": count})
    rows = pd.read_parquet(output)
    assert set(zip(rows["canvas_resolution"], rows["inter_px"], strict=True)) == {(8, 3), (16, 2)}
    assert len(rows) == 2 * 2 * 3
    assert json.loads(output.with_suffix(".json").read_text()) == {"8": {"count": 3}, "16": {"count": 2}}
