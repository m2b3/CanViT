import logging
import time
from pathlib import Path

import numpy as np
import polars as pl
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

from canvit_paper_exporter import paths
from canvit_paper_exporter.core import Dataset

log = logging.getLogger(__name__)


FRAC = 0.25
IT = 0
N_BOOT = 1000
N_GRID = 300
CI = 0.95


def _lowess_ci(
    x: np.ndarray, y: np.ndarray, x_grid: np.ndarray,
    *, frac: float, it: int, n_boot: int, ci: float, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta = 0.005 * (x_grid[-1] - x_grid[0])

    def _fit(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        out = sm_lowess(ys, xs, frac=frac, it=it, delta=delta, return_sorted=True, missing="drop")
        return np.interp(x_grid, out[:, 0], out[:, 1])

    smooth = _fit(x, y)
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, len(x_grid)))
    for b in range(n_boot):
        idx = rng.integers(len(x), size=len(x))
        boot[b] = _fit(x[idx], y[idx])
    alpha = (1 - ci) / 2
    return smooth, np.quantile(boot, alpha, axis=0), np.quantile(boot, 1 - alpha, axis=0)


def _group_curves(df: pl.DataFrame, group_col: str, x_grid: np.ndarray) -> list[dict]:
    rows = []
    for g in sorted(df[group_col].unique().to_list()):
        sub = df.filter(pl.col(group_col) == g)
        t0 = time.perf_counter()
        smooth, lo, hi = _lowess_ci(
            sub["area"].to_numpy(), sub["iou"].to_numpy(), x_grid,
            frac=FRAC, it=IT, n_boot=N_BOOT, ci=CI, seed=int(g),
        )
        log.info("  lowess %s=%d (n=%d): %.1fs", group_col, g, len(sub), time.perf_counter() - t0)
        rows.append({
            group_col: int(g),
            "smooth": smooth.tolist(),
            "ci_lo": lo.tolist(),
            "ci_hi": hi.tolist(),
        })
    return rows


def _delta_curves(
    df: pl.DataFrame, *,
    pair_cols: list[str], timestep_col: str,
    t_early: int, t_late: int,
    group_col: str, x_grid: np.ndarray,
) -> list[dict]:
    """LOWESS of Δiou = iou(t_late) - iou(t_early) vs area, grouped by `group_col`.

    Self-joins df on `pair_cols` between the two timesteps. Area is timestep-invariant,
    so we take it from the t_early snapshot.
    """
    early = (
        df.filter(pl.col(timestep_col) == t_early)
          .select(pair_cols + ["area", "iou"])
          .rename({"iou": "iou_early"})
    )
    late = (
        df.filter(pl.col(timestep_col) == t_late)
          .select(pair_cols + ["iou"])
          .rename({"iou": "iou_late"})
    )
    paired = (
        early.join(late, on=pair_cols, how="inner")
             .with_columns((pl.col("iou_late") - pl.col("iou_early")).alias("delta_iou"))
             .filter(pl.col("delta_iou").is_not_null() & pl.col("delta_iou").is_not_nan())
    )
    rows = []
    for g in sorted(paired[group_col].unique().to_list()):
        sub = paired.filter(pl.col(group_col) == g)
        smooth, lo, hi = _lowess_ci(
            sub["area"].to_numpy(), sub["delta_iou"].to_numpy(), x_grid,
            frac=FRAC, it=IT, n_boot=N_BOOT, ci=CI, seed=int(g),
        )
        rows.append({
            group_col: int(g),
            "smooth": smooth.tolist(),
            "ci_lo": lo.tolist(),
            "ci_hi": hi.tolist(),
        })
    return rows


def _load_obj_parquet(path: Path) -> pl.DataFrame:
    """Filter to GT-present rows and derive iou + area inline.

    Producer-side parquet contains the full N×C cross-product per
    (resolution|canvas, timestep). We filter `gt_area_px > 0` here
    (per-mask analyses only) and derive `area = gt_area_px / mask_resolution_px²`
    instead of relying on a separately-computed `area` column.
    """
    return (pl.read_parquet(path)
              .filter(pl.col("gt_area_px") > 0)
              .with_columns([
                  (pl.col("inter_px") / pl.col("union_px")).alias("iou"),
                  (pl.col("gt_area_px").cast(pl.Float64)
                   / (pl.col("mask_resolution_px").cast(pl.Float64) ** 2)).alias("area"),
              ]))


def compute() -> dict:
    obj_dir = paths.eval_dir("ade20k_obj")
    dv3_path = obj_dir / "dv3_iou.parquet"
    canvit_path = obj_dir / "canvit_iou.parquet"
    for p in (dv3_path, canvit_path):
        assert p.exists(), f"Missing {p}"

    dv3 = _load_obj_parquet(dv3_path)
    canvit_all = _load_obj_parquet(canvit_path)

    t_early = 0
    t_late = int(canvit_all["timestep"].max())
    canvit_t0 = canvit_all.filter(pl.col("timestep") == t_early)

    x_min = float(pl.concat([dv3["area"], canvit_t0["area"]]).min())
    x_max = float(pl.concat([dv3["area"], canvit_t0["area"]]).max())
    x_grid = np.linspace(x_min, x_max, N_GRID)

    return {
        "lowess": {"frac": FRAC, "it": IT, "n_boot": N_BOOT, "n_grid": N_GRID, "ci": CI},
        "plotting": {"t_delta_early": t_early, "t_delta_late": t_late},
        "x_grid": x_grid.tolist(),
        "dv3": _group_curves(dv3, "resolution", x_grid),
        "canvit_t0": _group_curves(canvit_t0, "canvas_resolution", x_grid),
        "canvit_delta": _delta_curves(
            canvit_all,
            pair_cols=["image_idx", "class_idx", "canvas_resolution"],
            timestep_col="timestep",
            t_early=t_early, t_late=t_late,
            group_col="canvas_resolution", x_grid=x_grid,
        ),
    }


dataset = Dataset(name="ade20k_iou_vs_obj", compute=compute)
