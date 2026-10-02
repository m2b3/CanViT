import logging

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure as MplFigure
from matplotlib.ticker import FuncFormatter

from canvit_paper_exporter import paths
from canvit_paper_exporter.core import Figure, load_json
from canvit_paper_exporter.style import (
    CI_ALPHA, LABEL_SIZE, LEGEND_SIZE, LINE_ALPHA, LINE_WIDTH,
    POLICY_ORDER, POLICY_STYLES,
    bottom_legend_with_header, inline_top_legend, linestyle_legend_handles,
    si_flops_label, three_panel_figure,
)

log = logging.getLogger(__name__)

PAPER_CONFIGS: set[tuple[int, int]] = {(512, 32), (512, 64)}

GRID_LINESTYLE = {32: "--", 64: "-"}
GRID_LABEL     = {32: "32² canvas", 64: "64² canvas"}

MODE_LINESTYLE = {"frozen": "--", "finetuned": "-"}
MODE_LABEL     = {"frozen": "Frozen", "finetuned": "Finetuned"}

FRONTIER_POLICIES = {"entropy_coarse_to_fine", "fine_to_coarse"}

# Distinct shape + color per baseline so the three are immediately distinguishable
# in panel A's boxed legend. Colors kept dark/neutral to avoid clashing with the
# CanViT policy palette.
BASELINE_STYLES: dict[str, dict] = {
    # Both AME variants as triangles (same family, different init), different colors
    # so you can tell them apart. AdaGlimpse as diamond (distinct family).
    "AME (SETR)":  {"marker": "^", "color": "#7f2704"},  # triangle-up, dark red-brown
    "AME (MAE)":   {"marker": "^", "color": "#00a4b8"},  # triangle-up, cyan
    "AdaGlimpse":  {"marker": "D", "color": "#6a3d9a"},  # diamond,      deep purple (distinct from red-brown AME)
}
BASELINE_MARKER_SIZE = 28


def _plot_mean_and_ci(ax: Axes, x, pts: list[dict], *, color: str, linestyle: str, n_runs: int) -> None:
    """Plot mean%+CI for a per-timestep entry. Scales mean/ci_lo/ci_hi by 100
    (every pipeline emits fractional values; paper renders as percent)."""
    mean = np.array([p["mean"] for p in pts]) * 100
    ax.plot(x, mean, color=color, linewidth=LINE_WIDTH, linestyle=linestyle, alpha=LINE_ALPHA)
    if n_runs >= 2:
        lo = np.array([p["ci_lo"] for p in pts]) * 100
        hi = np.array([p["ci_hi"] for p in pts]) * 100
        ax.fill_between(x, lo, hi, color=color, alpha=CI_ALPHA)


def _panel_frontier(ax: Axes, ade20k: dict) -> None:
    drawn = 0
    for entry in ade20k["policy_curves"]:
        if entry["policy"] not in FRONTIER_POLICIES:
            continue
        if (entry["scene_size"], entry["canvas_grid"]) not in PAPER_CONFIGS:
            continue
        style = POLICY_STYLES[entry["policy"]]
        pts = entry["per_timestep"]
        flops = np.array([p["cum_gflops"] for p in pts]) * 1e9
        _plot_mean_and_ci(ax, flops, pts, color=style["color"],
                          linestyle=GRID_LINESTYLE[entry["canvas_grid"]],
                          n_runs=entry["n_runs"])
        drawn += 1
    log.info("panel A: %d CanViT curves", drawn)

    # All baseline markers drawn, one legend entry per distinct `name` (so
    # AdaGlimpse T=4 + T=8 share one entry — matches the original layout).
    # Each baseline family gets a distinct marker + color (see BASELINE_STYLES).
    seen_labels: set[str] = set()
    baseline_handles = []
    for b in ade20k.get("baselines", []):
        style = BASELINE_STYLES.get(b["name"], {"marker": "x", "color": "k"})
        handle = ax.scatter(
            [b["gflops"] * 1e9], [b["miou_pct"]],
            marker=style["marker"], s=BASELINE_MARKER_SIZE, color=style["color"],
            zorder=5,
            label=b["name"] if b["name"] not in seen_labels else None,
        )
        if b["name"] not in seen_labels:
            baseline_handles.append(handle)
            seen_labels.add(b["name"])
    log.info("panel A: %d baseline markers, %d distinct legend entries",
             len(ade20k.get("baselines", [])), len(baseline_handles))

    ax.set_xlabel("Cumulative FLOPs", fontsize=LABEL_SIZE)
    ax.set_ylabel("ADE20K mIoU", fontsize=LABEL_SIZE)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=ax.get_xlim()[1] * 1.15)
    ax.xaxis.set_major_formatter(FuncFormatter(si_flops_label))

    inline_top_legend(ax, linestyle_legend_handles(GRID_LINESTYLE, GRID_LABEL))
    if baseline_handles:
        bl = ax.legend(
            handles=baseline_handles, fontsize=LEGEND_SIZE,
            loc="lower right", ncol=1, frameon=True,
            handletextpad=0.4, borderpad=0.45, markerscale=0.9,
        )
        ax.add_artist(bl)


def _panel_ade20k_policies(ax: Axes, ade20k: dict) -> None:
    max_t = 0
    drawn = 0
    for entry in ade20k["policy_curves"]:
        if entry["policy"] not in POLICY_STYLES:
            continue
        if (entry["scene_size"], entry["canvas_grid"]) not in PAPER_CONFIGS:
            continue
        style = POLICY_STYLES[entry["policy"]]
        pts = entry["per_timestep"]
        T = len(pts)
        max_t = max(max_t, T)
        _plot_mean_and_ci(ax, np.arange(T), pts, color=style["color"],
                          linestyle=GRID_LINESTYLE[entry["canvas_grid"]],
                          n_runs=entry["n_runs"])
        drawn += 1
    log.info("panel B: %d curves across %d timesteps", drawn, max_t)

    ax.set_xlabel("Timestep $t$", fontsize=LABEL_SIZE)
    ax.set_ylabel("ADE20K mIoU", fontsize=LABEL_SIZE)
    ax.set_ylim(bottom=38, top=47)
    ax.set_xticks(np.arange(0, max_t, step=5))

    inline_top_legend(ax, linestyle_legend_handles(GRID_LINESTYLE, GRID_LABEL))


def _plot_in1k_mode(ax: Axes, data: dict, linestyle: str, mode_name: str) -> None:
    drawn = 0
    for entry in data.get("configs", []):
        style = POLICY_STYLES.get(entry["policy"])
        if style is None:
            continue
        pts = entry["per_timestep"]
        _plot_mean_and_ci(ax, np.arange(len(pts)), pts, color=style["color"],
                          linestyle=linestyle, n_runs=entry["n_runs"])
        drawn += 1
    log.info("panel C [%s]: %d curves", mode_name, drawn)


def _panel_in1k(ax: Axes, frozen: dict, finetuned: dict) -> None:
    _plot_in1k_mode(ax, frozen, MODE_LINESTYLE["frozen"], "frozen")
    _plot_in1k_mode(ax, finetuned, MODE_LINESTYLE["finetuned"], "finetuned")

    ax.set_xlabel("Timestep $t$", fontsize=LABEL_SIZE)
    ax.set_ylabel("Top-1 accuracy", fontsize=LABEL_SIZE)
    ax.set_ylim(bottom=74, top=87)

    Ts = [len(c["per_timestep"]) for c in frozen.get("configs", [])]
    Ts += [len(c["per_timestep"]) for c in finetuned.get("configs", [])]
    if Ts:
        ax.set_xticks(np.arange(0, max(Ts), step=5))

    inline_top_legend(ax, linestyle_legend_handles(MODE_LINESTYLE, MODE_LABEL))


def plot() -> MplFigure:
    ade20k = load_json(paths.export_json("ade20k_seg"))
    frozen = load_json(paths.export_json("in1k_clf_frozen"))
    finetuned = load_json(paths.export_json("in1k_clf_finetuned"))
    log.info("ade20k_seg: %d policy_curves; in1k frozen=%d configs; in1k finetuned=%d configs",
             len(ade20k["policy_curves"]),
             len(frozen.get("configs", [])),
             len(finetuned.get("configs", [])))

    fig, axs = three_panel_figure()
    _panel_frontier(axs[0], ade20k)
    _panel_ade20k_policies(axs[1], ade20k)
    _panel_in1k(axs[2], frozen, finetuned)

    bottom_legend_with_header(
        fig, header="CanViT policy:",
        entries=[(POLICY_STYLES[p]["color"], POLICY_STYLES[p]["label"]) for p in POLICY_ORDER],
    )
    return fig


figure = Figure(name="main_results", plot=plot)
