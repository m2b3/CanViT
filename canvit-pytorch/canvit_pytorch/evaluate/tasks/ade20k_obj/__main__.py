"""ADE20K mask-analysis pipeline: DINOv3 feature export → DINOv3 IoU → CanViT IoU.
Each stage skips if its output exists; delete the artifact to force a rerun."""

import datetime as dt
import logging
from pathlib import Path

from canvit_eval.tasks.ade20k_obj.export_dv3_features import (
    ExportFeaturesConfig,
    main as export_dv3_features,
)
from canvit_eval.tasks.ade20k_obj.iou import (
    CANVIT_PROBE_REPOS,
    CanViTConfig,
    DINOv3Config,
    run_canvit,
    run_dinov3,
)
from canvit_eval.tasks.ade20k_obj.paths import (
    CANVIT_PARQUET,
    DV3_PARQUET,
    FEATURES_DIR,
    features_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


DV3_RESOLUTIONS_PX: list[int] = [128]
CANVAS_RESOLUTIONS: list[int] = [8, 16, 32, 64]


def _describe(path: Path) -> str:
    stat = path.stat()
    mtime = dt.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return f"{stat.st_size / 1e6:.1f} MB, mtime={mtime}"


def _skip_if_exists(path: Path, label: str) -> bool:
    if not path.exists():
        return False
    log.info("skip %s — %s already present (%s)", label, path, _describe(path))
    return True


assert set(CANVAS_RESOLUTIONS) <= CANVIT_PROBE_REPOS.keys(), (
    set(CANVAS_RESOLUTIONS) - CANVIT_PROBE_REPOS.keys()
)

log.info("=== ADE20K mask-analysis pipeline ===")
log.info("features dir:     %s", FEATURES_DIR)
log.info("dv3 iou parquet:  %s", DV3_PARQUET)
log.info("canvit iou parq:  %s", CANVIT_PARQUET)
log.info("dv3 resolutions:  %s", DV3_RESOLUTIONS_PX)
log.info("canvas resolutions: %s", CANVAS_RESOLUTIONS)

log.info("── DINOv3 features ──")
for res_px in DV3_RESOLUTIONS_PX:
    out = features_path(res_px)
    if _skip_if_exists(out, f"dv3 features @ {res_px}px"):
        continue
    export_dv3_features(ExportFeaturesConfig(eval_resolution_px=res_px))

log.info("── DINOv3 per-(image, class) IoU ──")
if not _skip_if_exists(DV3_PARQUET, "dv3 iou parquet"):
    run_dinov3(DINOv3Config(resolutions_px=DV3_RESOLUTIONS_PX))

log.info("── CanViT per-(image, class, timestep) IoU ──")
if not _skip_if_exists(CANVIT_PARQUET, "canvit iou parquet"):
    run_canvit(CanViTConfig(canvas_resolutions=CANVAS_RESOLUTIONS))

log.info("=== pipeline complete ===")
