import logging

import numpy as np
from canvit_pytorch.hub.repos import RELEASED_CANVAS_GRID_SIZE, RELEASED_SCENE_SIZE_PX
from matplotlib.axes import Axes
from matplotlib.figure import Figure as MplFigure
from matplotlib.ticker import FuncFormatter, MultipleLocator, NullLocator

from canvit_paper_exporter import in1k, learned_policies
from canvit_paper_exporter.ade20k import seg
from canvit_paper_exporter.core import Figure
from canvit_paper_exporter.style import (
    LABEL_SIZE, LEARNED_POLICY_STYLES, LEGEND_SIZE, LINE_ALPHA,
    POLICY_ORDER, POLICY_STYLES,
    bottom_legend_with_headers, inline_top_legend, linestyle_legend_handles, mean_and_ci,
    si_flops_label, two_by_two_figure,
)

log = logging.getLogger(__name__)

# (scene, canvas grid) of the curves drawn: the released configuration and the 64² inference canvas.
PAPER_CONFIGS = {(RELEASED_SCENE_SIZE_PX, RELEASED_CANVAS_GRID_SIZE), (RELEASED_SCENE_SIZE_PX, 64)}

GRID_LINESTYLE = {32: "--", 64: "-"}
GRID_LABEL     = {32: "32² canvas", 64: "64² canvas"}

MODE_LINESTYLE = {"frozen": "--", "finetuned": "-"}

# Panel D sets the policies apart by line style; its canvas is the caption's.
KIND_LINESTYLE = {"non_learned": "--", "learned": "-"}
KIND_LABEL     = {"non_learned": "Non-learned", "learned": "Learned"}
MODE_LABEL     = {"frozen": "Frozen", "finetuned": "Fine-tuned"}

FRONTIER_POLICIES = {"entropy_coarse_to_fine", "fine_to_coarse"}

# A marker shape per model family (the AME variants share one) and a color per baseline, apart from the policies' colors.
BASELINE_STYLES: dict[str, dict] = {
    "AME (SETR)":  {"marker": "^", "color": "#7f2704"},
    "AME (MAE)":   {"marker": "^", "color": "#00a4b8"},
    "AdaGlimpse":  {"marker": "D", "color": "#6a3d9a"},
}
BASELINE_MARKER_SIZE = 28

# Panels B and D share one mIoU axis, so that D reads as a zoom on B's first glimpses: R-IID and F2C, which start away
# from the full view, lie below its floor in both. The top stops short of 47 so that no tick label sits beside a panel
# letter.
ADE20K_POLICY_YLIM = (38, 46.5)


def _ade20k_policy_axis(ax: Axes, highest: float, panel: str) -> None:
    """Panels B and D's y axis: ADE20K_POLICY_YLIM, with a labeled tick at every mIoU point; `highest` is the panel's
    highest interval bound, in mIoU points."""
    assert highest < ADE20K_POLICY_YLIM[1], f"panel {panel}'s y limit {ADE20K_POLICY_YLIM[1]} cuts a curve at {highest:.2f}"
    ax.set_ylim(*ADE20K_POLICY_YLIM)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(NullLocator())


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
    highest = max(p["ci_hi"] for e in ade20k["policy_curves"] if (e["scene_size"], e["canvas_grid"]) in PAPER_CONFIGS
                  for p in e["per_timestep"]) * 100
    _ade20k_policy_axis(ax, highest, "B")
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
    ax.set_yticks([75, 80, 85])

    Ts = [len(c["per_timestep"]) for c in [*frozen["configs"], *finetuned["configs"]]]
    ax.set_xticks(np.arange(0, max(Ts), step=5))

    inline_top_legend(ax, linestyle_legend_handles(MODE_LINESTYLE, MODE_LABEL))



def _panel_learned(ax: Axes, ade20k: dict, learned: dict) -> None:
    """Panel B's 64² curves over the learned policies' episodes, every non-learned policy drawn from the same results as
    panel B (dashed), then the learned policies (solid)."""
    recipe = learned["recipe"]
    grid = recipe["canvas_grid"]
    assert (recipe["scene_size"], grid) == (RELEASED_SCENE_SIZE_PX, 64), "panel D assumes panel B's 64² curves"
    num_glimpses = recipe["num_glimpses"]
    non_learned = {
        c["policy"]: c for c in ade20k["policy_curves"] if (c["scene_size"], c["canvas_grid"]) == (recipe["scene_size"], grid)
    }
    # The learned-policy dataset carries its own copy of the non-learned policies it is compared with: every statistic it records
    # must equal panel B's (panel B's records also carry cumulative FLOPs).
    for c in learned["baselines"]:
        ref = non_learned[c["policy"]]
        same = c["n_runs"] == ref["n_runs"] and all(
            point.items() <= ref_point.items()
            for point, ref_point in zip(c["per_timestep"][:num_glimpses], ref["per_timestep"][:num_glimpses], strict=True)
        )
        assert same, f"{c['policy']}: the learned-policy dataset's curve differs from panel B's"
    timesteps = np.arange(num_glimpses)
    for policy in POLICY_ORDER:
        c = non_learned[policy]
        mean_and_ci(ax, timesteps, c["per_timestep"][:num_glimpses], n_runs=c["n_runs"],
                    color=POLICY_STYLES[policy]["color"], alpha=LINE_ALPHA, marker="o", markersize=2.5,
                    linestyle=KIND_LINESTYLE["non_learned"])
    plotted = [c for c in learned["learned"] if c["selection"] == learned_policies.PLOTTED_SELECTION]
    for c in plotted:
        mean_and_ci(ax, timesteps, c["per_timestep"], n_runs=len(c["seeds"]),
                    color=LEARNED_POLICY_STYLES[c["objective"]]["color"], alpha=LINE_ALPHA, marker="o", markersize=2.5,
                    linestyle=KIND_LINESTYLE["learned"])
    highest = max(p["ci_hi"] for c in [*plotted, *non_learned.values()] for p in c["per_timestep"][:num_glimpses]) * 100
    _ade20k_policy_axis(ax, highest, "D")
    ax.set_xticks(timesteps)
    ax.xaxis.set_minor_locator(NullLocator())  # timesteps are whole glimpses
    ax.set_xlabel("Timestep $t$", fontsize=LABEL_SIZE)
    ax.set_ylabel("ADE20K mIoU", fontsize=LABEL_SIZE)
    inline_top_legend(ax, linestyle_legend_handles(KIND_LINESTYLE, KIND_LABEL))


def plot(ade20k: dict, frozen: dict, finetuned: dict, learned: dict) -> MplFigure:
    log.info("ade20k_seg: %d policy_curves; in1k frozen=%d configs; in1k finetuned=%d configs",
             len(ade20k["policy_curves"]),
             len(frozen["configs"]),
             len(finetuned["configs"]))

    fig, axs = two_by_two_figure()
    _panel_frontier(axs[0], ade20k)
    _panel_ade20k_policies(axs[1], ade20k)
    _panel_in1k(axs[2], frozen, finetuned)
    _panel_learned(axs[3], ade20k, learned)

    bottom_legend_with_headers(fig, [
        ("Non-learned policy:", [(POLICY_STYLES[p]["color"], POLICY_STYLES[p]["label"], "-") for p in POLICY_ORDER]),
        ("Learned policy:", [(s["color"], s["label"], "-") for s in LEARNED_POLICY_STYLES.values()]),
    ], rows=True)
    return fig


figure = Figure(
    name="main_results", plot=plot,
    datasets=(seg.dataset, in1k.frozen_dataset, in1k.finetuned_dataset, learned_policies.dataset),
)
