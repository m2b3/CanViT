import logging

import numpy as np
from canvit_pytorch.hub.repos import RELEASED_CANVAS_GRID_SIZE, RELEASED_SCENE_SIZE_PX
from matplotlib.axes import Axes
from matplotlib.figure import Figure as MplFigure
from matplotlib.ticker import FuncFormatter

from canvit_paper_exporter import in1k
from canvit_paper_exporter.ade20k import seg
from canvit_paper_exporter.core import Figure
from canvit_paper_exporter.style import (
    LABEL_SIZE, LEGEND_SIZE, LINE_ALPHA,
    POLICY_ORDER, POLICY_STYLES,
    bottom_legend_with_header, inline_top_legend, linestyle_legend_handles, mean_and_ci,
    si_flops_label, three_panel_figure,
)

log = logging.getLogger(__name__)

# (scene, canvas grid) of the curves drawn: the released configuration and the 64² inference canvas.
PAPER_CONFIGS = {(RELEASED_SCENE_SIZE_PX, RELEASED_CANVAS_GRID_SIZE), (RELEASED_SCENE_SIZE_PX, 64)}

GRID_LINESTYLE = {32: "--", 64: "-"}
GRID_LABEL     = {32: "32² canvas", 64: "64² canvas"}

MODE_LINESTYLE = {"frozen": "--", "finetuned": "-"}
MODE_LABEL     = {"frozen": "Frozen", "finetuned": "Finetuned"}

FRONTIER_POLICIES = {"entropy_coarse_to_fine", "fine_to_coarse"}

# A marker shape per model family (the AME variants share one) and a color per baseline, apart from the policies' colors.
BASELINE_STYLES: dict[str, dict] = {
    "AME (SETR)":  {"marker": "^", "color": "#7f2704"},
    "AME (MAE)":   {"marker": "^", "color": "#00a4b8"},
    "AdaGlimpse":  {"marker": "D", "color": "#6a3d9a"},
}
BASELINE_MARKER_SIZE = 28


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
        mean_and_ci(ax, flops, pts, n_runs=entry["n_runs"], color=style["color"],
                    linestyle=GRID_LINESTYLE[entry["canvas_grid"]], alpha=LINE_ALPHA)
        drawn += 1
    log.info("panel A: %d CanViT curves", drawn)

    # A legend entry per baseline name: AdaGlimpse's two glimpse counts share one.
    seen_labels: set[str] = set()
    baseline_handles = []
    for b in ade20k["baselines"]:
        style = BASELINE_STYLES[b["name"]]
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
             len(ade20k["baselines"]), len(baseline_handles))

    ax.set_xlabel("Cumulative FLOPs", fontsize=LABEL_SIZE)
    ax.set_ylabel("ADE20K mIoU", fontsize=LABEL_SIZE)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=ax.get_xlim()[1] * 1.15)
    ax.xaxis.set_major_formatter(FuncFormatter(si_flops_label))

    inline_top_legend(ax, linestyle_legend_handles(GRID_LINESTYLE, GRID_LABEL))
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
        if (entry["scene_size"], entry["canvas_grid"]) not in PAPER_CONFIGS:
            continue
        style = POLICY_STYLES[entry["policy"]]
        pts = entry["per_timestep"]
        T = len(pts)
        max_t = max(max_t, T)
        mean_and_ci(ax, np.arange(T), pts, n_runs=entry["n_runs"], color=style["color"],
                    linestyle=GRID_LINESTYLE[entry["canvas_grid"]], alpha=LINE_ALPHA)
        drawn += 1
    log.info("panel B: %d curves across %d timesteps", drawn, max_t)

    ax.set_xlabel("Timestep $t$", fontsize=LABEL_SIZE)
    ax.set_ylabel("ADE20K mIoU", fontsize=LABEL_SIZE)
    ax.set_ylim(bottom=38, top=47)
    ax.set_xticks(np.arange(0, max_t, step=5))

    inline_top_legend(ax, linestyle_legend_handles(GRID_LINESTYLE, GRID_LABEL))


def _plot_in1k_mode(ax: Axes, data: dict, linestyle: str, mode_name: str) -> None:
    drawn = 0
    for entry in data["configs"]:
        style = POLICY_STYLES[entry["policy"]]
        pts = entry["per_timestep"]
        mean_and_ci(ax, np.arange(len(pts)), pts, n_runs=entry["n_runs"], color=style["color"],
                    linestyle=linestyle, alpha=LINE_ALPHA)
        drawn += 1
    log.info("panel C [%s]: %d curves", mode_name, drawn)


def _panel_in1k(ax: Axes, frozen: dict, finetuned: dict) -> None:
    _plot_in1k_mode(ax, frozen, MODE_LINESTYLE["frozen"], "frozen")
    _plot_in1k_mode(ax, finetuned, MODE_LINESTYLE["finetuned"], "finetuned")

    ax.set_xlabel("Timestep $t$", fontsize=LABEL_SIZE)
    ax.set_ylabel("Top-1 accuracy", fontsize=LABEL_SIZE)
    ax.set_ylim(bottom=74, top=87)

    Ts = [len(c["per_timestep"]) for c in [*frozen["configs"], *finetuned["configs"]]]
    ax.set_xticks(np.arange(0, max(Ts), step=5))

    inline_top_legend(ax, linestyle_legend_handles(MODE_LINESTYLE, MODE_LABEL))


def plot(ade20k: dict, frozen: dict, finetuned: dict) -> MplFigure:
    log.info("ade20k_seg: %d policy_curves; in1k frozen=%d configs; in1k finetuned=%d configs",
             len(ade20k["policy_curves"]),
             len(frozen["configs"]),
             len(finetuned["configs"]))

    fig, axs = three_panel_figure()
    _panel_frontier(axs[0], ade20k)
    _panel_ade20k_policies(axs[1], ade20k)
    _panel_in1k(axs[2], frozen, finetuned)

    bottom_legend_with_header(
        fig, header="CanViT policy:",
        entries=[(POLICY_STYLES[p]["color"], POLICY_STYLES[p]["label"]) for p in POLICY_ORDER],
    )
    return fig


figure = Figure(
    name="main_results", plot=plot, datasets=(seg.dataset, in1k.frozen_dataset, in1k.finetuned_dataset),
)
