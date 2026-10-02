import json
from typing import NamedTuple

import numpy as np
import ultraplot as uplt
from matplotlib.figure import Figure as MplFigure
from matplotlib.ticker import FuncFormatter

from canvit_paper_exporter import paths
from canvit_paper_exporter.ablations.registry import BASELINE, BY_SLUG
from canvit_paper_exporter.core import Figure, load_json


# Row × column layout. Each cell plots one (policy, metric) series.
ROW_POLICIES = [("random", "R-IID Policy"), ("full", "F-IID Policy")]
COL_METRICS  = [("patches_loss", "Patch MSE"), ("cls_loss", "CLS MSE")]

# TensorBoard-style single-pole EMA; decay=0.99 gives a ~100-step effective window.
EMA_DECAY = 0.99

# Skip pretraining warmup so the informative variation isn't compressed at the left edge.
# Matches the baseline's warmup_steps.
MIN_STEP = 20_000

LW_DEFAULT  = 1.0
LW_BASELINE = 1.3
LW_RAW      = 0.15
ALPHA_SMOOTH = 0.75
ALPHA_RAW    = 0.10


class Series(NamedTuple):
    steps: np.ndarray
    raw: np.ndarray
    smoothed: np.ndarray


def _ema(values: np.ndarray, decay: float) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = (1 - decay) * values[i] + decay * out[i - 1]
    return out


def _prepare(points: list[dict], *, min_step: int, max_step: int, ema_decay: float) -> Series | None:
    if not points:
        return None
    steps = np.array([p["step"] for p in points])
    values = np.array([p["value"] for p in points])
    smoothed = _ema(values, ema_decay)
    mask = (steps >= min_step) & (steps <= max_step)
    if not mask.any():
        return None
    return Series(steps[mask], values[mask], smoothed[mask])


def _metric_key(policy: str, metric: str) -> str:
    return f"{policy}_{metric}"


def _load_curves() -> dict[str, dict]:
    """Load the exported Comet curves for every registered variant, keyed by slug."""
    out: dict[str, dict] = {}
    for slug in BY_SLUG:
        path = paths.comet_curves() / f"{slug}.json"
        assert path.exists(), f"Missing curves file: {path}"
        out[slug] = json.loads(path.read_text())
    return out


def _max_observed_step(curves: dict[str, dict]) -> int:
    return max(
        (series[-1]["step"] for v in curves.values() for series in v.values() if series),
        default=0,
    )


def _plot_panel(
    ax, curves: dict[str, dict], metric_key: str, order: list[str], variant_meta: dict,
    *, min_step: int, max_step: int, ema_decay: float,
) -> None:
    ymin, ymax = float("inf"), float("-inf")
    for slug in order:
        meta = variant_meta[slug]
        series = _prepare(
            curves.get(slug, {}).get(metric_key, []),
            min_step=min_step, max_step=max_step, ema_decay=ema_decay,
        )
        if series is None:
            continue
        lw = LW_BASELINE if slug == BASELINE.slug else LW_DEFAULT
        ax.plot(series.steps, series.raw, color=meta["color"], linewidth=LW_RAW, alpha=ALPHA_RAW)
        ax.plot(series.steps, series.smoothed, color=meta["color"], linewidth=lw,
                alpha=ALPHA_SMOOTH, label=meta["label"])
        ymin = min(ymin, float(series.smoothed.min()))
        ymax = max(ymax, float(series.smoothed.max()))
    if ymin != float("inf"):
        margin = 0.05 * (ymax - ymin)
        ax.set_ylim(ymin - margin, ymax + margin)
    ax.set_xlim(min_step, max_step)
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, _pos: f"{x / 1000:.0f}k" if x >= 1000 else f"{x:.0f}"
    ))


def plot() -> MplFigure:
    ablations = load_json(paths.export_json("ablation_variants"))
    curves = _load_curves()
    order = ablations["_meta"]["order"]
    max_step = _max_observed_step(curves)

    fig, axs = uplt.subplots(
        ncols=len(COL_METRICS), nrows=len(ROW_POLICIES),
        figheight=11 / 2.54, refaspect=1, sharex=False, sharey=False,
    )
    axs.format(
        toplabels=[col_label for _, col_label in COL_METRICS],
        toplabelsize=10, toplabelweight="bold",
        leftlabels=[row_label for _, row_label in ROW_POLICIES],
        leftlabelsize=10, leftlabelweight="bold",
        grid=False,
    )
    for ax in axs:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for r, (policy, _) in enumerate(ROW_POLICIES):
        for c, (metric, _) in enumerate(COL_METRICS):
            _plot_panel(
                axs[r * len(COL_METRICS) + c], curves,
                _metric_key(policy, metric), order, ablations["variants"],
                min_step=MIN_STEP, max_step=max_step, ema_decay=EMA_DECAY,
            )
    axs.format(xlabel="Training step")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="right", ncol=1, fontsize=7, frame=False)
    return fig


figure = Figure(name="ablation_loss", plot=plot)
