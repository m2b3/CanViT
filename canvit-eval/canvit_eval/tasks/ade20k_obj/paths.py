"""Output paths for the ADE20K mask-size pipeline."""

from pathlib import Path


RESULTS_DIR = Path("./results")
TASK_DIR = RESULTS_DIR / "ade20k_obj"
FEATURES_DIR = TASK_DIR / "dv3_features"
DV3_PARQUET = TASK_DIR / "dv3_iou.parquet"
CANVIT_PARQUET = TASK_DIR / "canvit_iou.parquet"


def features_path(eval_resolution_px: int) -> Path:
    return FEATURES_DIR / f"{eval_resolution_px}px_features.pt"


EXPECTED_N_VAL_IMAGES = 2000
