"""Pretraining losses of the ablation runs by step, smoothed, from their Comet histories."""

from typing import NamedTuple

import numpy as np
import ultraplot as uplt
from canvit_pytorch.policies import POLICIES
from canvit_pytorch.pretrain.ablations import ABLATION_WARMUP_STEPS
from matplotlib.figure import Figure as MplFigure
from matplotlib.ticker import FuncFormatter

from canvit_paper_exporter.ablations import history, static
from canvit_paper_exporter.ablations.registry import BASELINE, BY_SLUG
from canvit_paper_exporter.core import Figure


# Row × column layout. Each cell plots one (pretraining rollout, metric) series; a row names the rollout's policy.
ROW_POLICIES = [("random", f"{POLICIES['random'].paper_name} Policy"),
                ("full", f"{POLICIES['full_then_random'].paper_name} Policy")]
COL_METRICS  = [("patches_loss", "Patch MSE"), ("cls_loss", "CLS MSE")]

# TensorBoard-style single-pole EMA over the logged points; decay=0.99 averages about the last 100.
EMA_DECAY = 0.99

# Past the warmup, so the informative variation isn't compressed at the left edge.
MIN_STEP = ABLATION_WARMUP_STEPS

LW_DEFAULT  = 1.0
LW_BASELINE = 1.3
LW_RAW      = 0.15
ALPHA_SMOOTH = 0.75
ALPHA_RAW    = 0.10

# A panel spans about 476 pixel columns in the paper at 300 dpi.
M4_BINS = 500

HISTORIES = history.runs(*BY_SLUG)


class Curve(NamedTuple):
    steps: np.ndarray
    raw: np.ndarray
    smoothed: np.ndarray


def _ema(values: np.ndarray, decay: float) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = (1 - decay) * values[i] + decay * out[i - 1]
    return out


def _curve(series: history.Series, *, min_step: int, max_step: int, ema_decay: float) -> Curve | None:
    if not len(series.steps):
        return None
    smoothed = _ema(series.values, ema_decay)
    mask = (series.steps >= min_step) & (series.steps <= max_step)
    if not mask.any():
        return None
    return Curve(series.steps[mask], series.values[mask], smoothed[mask])


def _m4(steps: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep the first, lowest, highest and last point of each step bin (M4 decimation).

    A line through these points covers the same pixels as the full line when a bin is no wider
    than a pixel column.
    """
    assert (np.diff(steps) > 0).all()
    b = (steps - steps[0]) * M4_BINS // (steps[-1] - steps[0] + 1)
    starts = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
    ends = np.r_[starts[1:], len(b)] - 1
    by_value = np.lexsort((values, b))
    keep = np.unique(np.concatenate([starts, ends, by_value[starts], by_value[ends]]))
    return steps[keep], values[keep]


def _metric_key(policy: str, metric: str) -> str:
    return f"{policy}_{metric}"


def _max_observed_step(histories: dict[str, dict[str, history.Series]]) -> int:
    return max(int(s.steps[-1]) for metrics in histories.values() for s in metrics.values() if len(s.steps))


def _plot_panel(
    ax, histories: dict[str, dict[str, history.Series]], metric_key: str, order: list[str], variant_meta: dict,
    *, min_step: int, max_step: int, ema_decay: float,
) -> None:
    ymin, ymax = float("inf"), float("-inf")
    for slug in order:
        meta = variant_meta[slug]
        curve = _curve(histories[slug][metric_key], min_step=min_step, max_step=max_step, ema_decay=ema_decay)
        if curve is None:
            continue
        lw = LW_BASELINE if slug == BASELINE.slug else LW_DEFAULT
        ax.plot(*_m4(curve.steps, curve.raw), color=meta["color"], linewidth=LW_RAW, alpha=ALPHA_RAW)
        ax.plot(*_m4(curve.steps, curve.smoothed), color=meta["color"], linewidth=lw,
                alpha=ALPHA_SMOOTH, label=meta["label"])
        ymin = min(ymin, float(curve.smoothed.min()))
        ymax = max(ymax, float(curve.smoothed.max()))
    if ymin != float("inf"):
        margin = 0.05 * (ymax - ymin)
        ax.set_ylim(ymin - margin, ymax + margin)
    ax.set_xlim(min_step, max_step)
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, _pos: f"{x / 1000:.0f}k" if x >= 1000 else f"{x:.0f}"
    ))


def plot(ablations: dict) -> MplFigure:
    histories = {slug: history.load(HISTORIES, slug) for slug in BY_SLUG}
    order = ablations["_meta"]["order"]
    max_step = _max_observed_step(histories)

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
                axs[r * len(COL_METRICS) + c], histories,
                _metric_key(policy, metric), order, ablations["variants"],
                min_step=MIN_STEP, max_step=max_step, ema_decay=EMA_DECAY,
            )
    axs.format(xlabel="Training step")

    # Some runs lack some panels (no dense loss, no F-IID branch); every run gets a legend entry.
    by_label = {label: h for ax in axs for h, label in zip(*ax.get_legend_handles_labels(), strict=True)}
    labels = [ablations["variants"][slug]["label"] for slug in order]
    fig.legend([by_label[label] for label in labels], labels, loc="right", ncol=1, fontsize=7, frame=False)
    return fig


figure = Figure(name="ablation_loss", plot=plot, datasets=(static.dataset,), inputs=(HISTORIES,))
