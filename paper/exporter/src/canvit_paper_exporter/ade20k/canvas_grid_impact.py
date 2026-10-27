import logging

import numpy as np
import ultraplot as uplt
from canvit_pytorch.hub.repos import RELEASED_SCENE_SIZE_PX
from matplotlib.figure import Figure as MplFigure

from canvit_paper_exporter.ade20k import seg
from canvit_paper_exporter.ade20k.mask_size import GRID_COLORS
from canvit_paper_exporter.core import Figure
from canvit_paper_exporter.style import (
    LABEL_SIZE, POLICY_STYLES, SINGLE_PANEL_FIGHEIGHT, SINGLE_PANEL_FIGWIDTH, TICK_SIZE, bottom_legend_with_header,
    mean_and_ci,
)

log = logging.getLogger(__name__)


POLICY = "coarse_to_fine"


def plot(data: dict) -> MplFigure:
    entries = [
        e for e in data["policy_curves"]
        if e["policy"] == POLICY
        and e["canvas_grid"] in GRID_COLORS
        and e["scene_size"] == RELEASED_SCENE_SIZE_PX
    ]
    entries.sort(key=lambda e: e["canvas_grid"])
    assert entries, f"no entries for policy={POLICY} scene_size={RELEASED_SCENE_SIZE_PX}"

    fig, axs = uplt.subplots(
        ncols=1, figwidth=SINGLE_PANEL_FIGWIDTH, figheight=SINGLE_PANEL_FIGHEIGHT, abc=False,
    )
    ax = axs[0]
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.2)

    for e in entries:
        ts = np.array([p["t"] for p in e["per_timestep"]])
        mean_and_ci(ax, ts, e["per_timestep"], n_runs=e["n_runs"], color=GRID_COLORS[e["canvas_grid"]],
                    marker="o", markersize=3)

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
             len(entries), [e["canvas_grid"] for e in entries], RELEASED_SCENE_SIZE_PX, POLICY)
    return fig


figure = Figure(name=f"canvas_grid_impact_{POLICY}", plot=plot, datasets=(seg.dataset,))
