"""Two-panel Canvas-Attention scaling figure.

Two stacked panels, each at textwidth / 3 column width:

  Panel B — Total FLOPs vs output grid for DINOv3 ViT-B/S (passive baseline,
            quadratic in the patch grid) overlaid against CanViT-B at T=1
            (single forward pass) and T=21 (full rollout). Capital T marks
            CUMULATIVE forward-pass count, not an instantaneous timestep.

  Panel C — FLOP ratio (full / asym.) per Canvas Attention Read/Write pair,
            sweeping canvas patch grid on x and glimpse patch grid as the
            colour family. Both axes are in TOKEN counts; legend reads as
            the glimpse patch grid `g²` so 8² → 64 patch tokens etc.

The diagram-side wrapper (`diagrams/canvas_attention_combined_standalone.typ`)
composes a separate LHS canvas_attention diagram alongside this two-panel SVG
into a single output SVG. Letters B / C (this file) anchor the panel structure;
the LHS contributes A.
"""

from dataclasses import replace

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as MplFigure
from matplotlib.ticker import FuncFormatter

from canvit_pytorch.flops import dinov3_flops

from canvit_paper_exporter.core import Figure
from canvit_paper_exporter.flops.canvit import CANVIT_B, read_write_pair_flops, segmentation_glimpse_flops
from canvit_paper_exporter.flops.dinov3 import DINOV3_CONFIGS, DINOV3_PATCH_SIZE
from canvit_paper_exporter.style import (
    FIGWIDTH, LABEL_SIZE, LEGEND_SIZE, LINE_WIDTH, TICK_SIZE, si_flops_label,
)


# ── panel B (DINOv3-vs-CanViT scalability) constants ────────────────────
SPATIAL_GRIDS = (8, 16, 32, 64, 128)
T_EARLY = 0   # one forward pass  (T = 1)
T_LATE = 20   # full rollout      (T = 21)
NEUTRAL = "0.15"
LS_DV3 = "--"
LS_CV = "-"
LW_THICK = LINE_WIDTH * 1.6
LW_THIN = LINE_WIDTH * 0.7

# Drop the "DINOv3 " prefix in panel B: the dashed line style alone encodes
# the model family, and at this column width the prefix dominates the label.
DINOV3_PREFIX = False

# ── panel C (FLOP-ratio) constants ──────────────────────────────────────
# Glimpse grids start at 8² (CanViT-B's actual operating point); the 4²
# extreme is dropped to keep the panel focused on the realistic regime.
GLIMPSE_GRIDS = (8, 16, 32)
CANVAS_GRIDS = (8, 16, 32, 64, 128)
GLIMPSE_COLORS = {8: "#1f77b4", 16: "#ff7f0e", 32: "#2ca02c"}


def _rw_pair_flops(glimpse_grid: int, canvas_grid: int, *, full_proj: bool) -> int:
    config = replace(CANVIT_B, canvas_projections="qkvo" if full_proj else "asymmetric")
    glimpse_size_px = glimpse_grid * CANVIT_B.backbone_spec.patch_size
    return read_write_pair_flops(config, glimpse_size_px=glimpse_size_px, canvas_grid_size=canvas_grid)


def _canvit_forward(grid: int) -> int:
    return segmentation_glimpse_flops(CANVIT_B, canvas_grid_size=grid)


def _dinov3(name: str, grid: int) -> int:
    return dinov3_flops(DINOV3_CONFIGS[name], input_size_px=grid * DINOV3_PATCH_SIZE)


def _panel_B(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    grids = list(range(SPATIAL_GRIDS[0], SPATIAL_GRIDS[-1] + 1))
    x = [g ** 2 for g in grids]
    ax.plot(x, [_dinov3("DINOv3 ViT-B/16", g) for g in grids],
            color=NEUTRAL, ls=LS_DV3, lw=LW_THICK)
    ax.plot(x, [_dinov3("DINOv3 ViT-S/16", g) for g in grids],
            color=NEUTRAL, ls=LS_DV3, lw=LW_THIN)
    ax.plot(x, [_canvit_forward(g) * (T_LATE + 1) for g in grids],
            color=NEUTRAL, ls=LS_CV, lw=LW_THICK)
    ax.plot(x, [_canvit_forward(g) * (T_EARLY + 1) for g in grids],
            color=NEUTRAL, ls=LS_CV, lw=LW_THIN)
    ax.set_xscale("log", base=2)
    ax.set_xticks([g ** 2 for g in SPATIAL_GRIDS])
    ax.set_xticklabels([rf"${g}^2$" for g in SPATIAL_GRIDS], fontsize=TICK_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(si_flops_label))
    ax.set_xlabel("Scene patches", fontsize=LABEL_SIZE)
    ax.set_ylabel("Total FLOPs", fontsize=LABEL_SIZE)
    dv3 = "DINOv3 " if DINOV3_PREFIX else ""
    handles = [
        mlines.Line2D([], [], color=NEUTRAL, lw=LW_THICK, ls=LS_DV3, label=f"{dv3}ViT-B"),
        mlines.Line2D([], [], color=NEUTRAL, lw=LW_THIN,  ls=LS_DV3, label=f"{dv3}ViT-S"),
        mlines.Line2D([], [], color=NEUTRAL, lw=LW_THICK, ls=LS_CV,  label=rf"CanViT-B ($T={T_LATE + 1}$)"),
        mlines.Line2D([], [], color=NEUTRAL, lw=LW_THIN,  ls=LS_CV,  label=rf"CanViT-B ($T={T_EARLY + 1}$)"),
    ]
    ax.legend(handles=handles,
              fontsize=LEGEND_SIZE - 0.5, ncol=2,
              frameon=True, framealpha=0.9, edgecolor="lightgray",
              loc="lower center", bbox_to_anchor=(0.5, 1.02),
              handlelength=2.6, handletextpad=0.4, columnspacing=0.6, borderpad=0.25)
    ax.set_box_aspect(1.0)


def _panel_C(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for lg in GLIMPSE_GRIDS:
        ratios = [_rw_pair_flops(lg, cg, full_proj=True)
                  / _rw_pair_flops(lg, cg, full_proj=False) for cg in CANVAS_GRIDS]
        ax.plot(CANVAS_GRIDS, ratios, marker="o", markersize=3,
                linewidth=LINE_WIDTH * 0.8, color=GLIMPSE_COLORS[lg])
    ax.set_xlabel("Scene/canvas patches", fontsize=LABEL_SIZE)
    ax.set_ylabel("FLOP ratio\n(full / asym.)", fontsize=LABEL_SIZE)
    ax.set_xscale("log", base=2)
    ax.set_xticks(CANVAS_GRIDS)
    ax.set_xticklabels([rf"${g}^2$" for g in CANVAS_GRIDS], fontsize=TICK_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    handles = [mlines.Line2D([], [], ls="", marker="o", markersize=4,
                             color=GLIMPSE_COLORS[lg],
                             label=rf"${lg}^2$")
               for lg in GLIMPSE_GRIDS]
    leg = ax.legend(handles=handles,
              fontsize=LEGEND_SIZE - 0.5, ncol=1,
              title="Glimpse patches",
              title_fontproperties={"weight": "bold", "size": LEGEND_SIZE - 0.5},
              frameon=False,
              loc="upper left",
              handletextpad=0.2, borderpad=0, borderaxespad=0.4)
    leg._legend_box.align = "left"
    ax.set_box_aspect(1.0)


def plot() -> MplFigure:
    w = FIGWIDTH * 3 / 8
    h = 2.73
    fig, (ax_B, ax_C) = plt.subplots(2, 1, figsize=(w, h))
    fig.subplots_adjust(left=0.32, right=0.97, top=0.97, bottom=0.10, hspace=0.20)
    _panel_B(ax_B)
    _panel_C(ax_C)
    # Axes-relative letters track each panel as set_box_aspect(1.0) reshapes
    # the axes box.
    for ax, letter in [(ax_B, "B"), (ax_C, "C")]:
        ax.text(-0.48, 1.20, letter, transform=ax.transAxes,
                va="top", ha="left", fontsize=12, fontweight="bold")
    return fig


figure = Figure(name="canvas_attention_rhs", plot=plot)
