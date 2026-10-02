import logging

import numpy as np
import ultraplot as uplt
from matplotlib.figure import Figure as MplFigure

from canvit_paper_exporter import paths
from canvit_paper_exporter.ade20k.mask_size import GRID_COLORS
from canvit_paper_exporter.comparison import POLICY_STYLES
from canvit_paper_exporter.core import Figure, load_json
from canvit_paper_exporter.style import (
    CI_ALPHA, LABEL_SIZE, LINE_WIDTH, TICK_SIZE,
    bottom_legend_with_header,
)

log = logging.getLogger(__name__)


POLICY = "coarse_to_fine"
SCENE_SIZE = 512

# Wider/shorter than the FIGWIDTH/REFASPECT defaults so this single-axis
# figure doesn't waste vertical space when included alongside a table.
FIGWIDTH_LOCAL = 4.95
FIGHEIGHT_LOCAL = 2.85


def plot() -> MplFigure:
    data = load_json(paths.export_json("ade20k_seg"))

    entries = [
        e for e in data["policy_curves"]
        if e["policy"] == POLICY
        and e["canvas_grid"] in GRID_COLORS
        and e["scene_size"] == SCENE_SIZE
    ]
    entries.sort(key=lambda e: e["canvas_grid"])
    assert entries, f"no entries for policy={POLICY} scene_size={SCENE_SIZE}"

    fig, axs = uplt.subplots(
        ncols=1, figwidth=FIGWIDTH_LOCAL, figheight=FIGHEIGHT_LOCAL, abc=False,
    )
    ax = axs[0]
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.2)

    for e in entries:
        cg = e["canvas_grid"]
        ts = np.array([p["t"] for p in e["per_timestep"]])
        mean = np.array([p["mean"] for p in e["per_timestep"]]) * 100
        lo = np.array([p["ci_lo"] for p in e["per_timestep"]]) * 100
        hi = np.array([p["ci_hi"] for p in e["per_timestep"]]) * 100
        ax.plot(ts, mean, marker="o", markersize=3, linewidth=LINE_WIDTH,
                color=GRID_COLORS[cg])
        if e["n_runs"] >= 2:
            ax.fill_between(ts, lo, hi, color=GRID_COLORS[cg], alpha=CI_ALPHA)

    max_t = max(len(e["per_timestep"]) for e in entries)
    ax.set_xticks(np.arange(0, max_t, step=5))
    ax.set_xlabel("Timestep $t$", fontsize=LABEL_SIZE)
    ax.set_ylabel("ADE20K mIoU (%)", fontsize=LABEL_SIZE)
    ax.set_title(POLICY_STYLES[POLICY]["label"], fontsize=LABEL_SIZE + 1)

    bottom_legend_with_header(
        fig, header="Output patch grid:",
        entries=[(GRID_COLORS[e["canvas_grid"]], rf"${e['canvas_grid']}^2$") for e in entries],
    )
    log.info("canvas_grid_impact: %d curves (grids=%s, scene=%d, policy=%s)",
             len(entries), [e["canvas_grid"] for e in entries], SCENE_SIZE, POLICY)
    return fig


# Name includes the policy suffix so the rendered SVG has a unique,
# policy-specific filename. If the figure gains additional policies in
# the future, switch back to a plain `canvas_grid_impact` stem and emit
# one SVG per policy.
figure = Figure(name=f"canvas_grid_impact_{POLICY}", plot=plot)
