import logging

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure as MplFigure
from matplotlib.ticker import FuncFormatter

from canvit_paper_exporter import paths
from canvit_paper_exporter.core import Figure, load_json
from canvit_paper_exporter.style import (
    CI_ALPHA, LABEL_SIZE, LEGEND_SIZE, LINE_ALPHA, LINE_WIDTH,
    bottom_legend_with_header, linestyle_legend_handles,
    si_flops_label, three_panel_figure,
)

log = logging.getLogger(__name__)


# Same hue across panels A / B / C for a given output grid. Only grids that
# actually appear on a panel — no dead-weight legend swatches.
GRID_COLORS: dict[int, str] = {
    8:  "#0F245C",    # navy
    16: "#0F7C83",    # teal
    32: "#B83292",    # magenta
    64: "#EF7A1B",    # orange
}

LS_DV3 = "--"
LS_CV = "-"
NEUTRAL = "0.15"

DV3_PATCH_SIZE = 16             # DINOv3 ViT-B/16 patch size: input_px / 16 = output grid
FRONTIER_POLICY = "entropy_coarse_to_fine"

# Shared convention for the linestyle inline legend on panels A and C.
MODEL_LINESTYLE = {"DINOv3 ViT-B": LS_DV3, "CanViT-B": LS_CV}

# Panel C (Pareto) x-axis range. Log-scale — at linear the leftmost curve
# collapses onto the y-axis and the DINOv3 annotations cluster at the right.
PARETO_FLOPS_MIN = 9e9        # below the smallest CanViT c=8 cum_gflops at t=0
PARETO_XLIM_RIGHT_PAD = 3.25    # leaves room for the "512² px" annotation tail

# IoU / Δ panels share a log-scale x-axis from 0.1% to 100% scene area.
X_LOG_MIN = 1e-3

# DINOv3 ViT-B probes the Pareto panel overlays as scatter + label.
# The three powers-of-two inputs map cleanly onto the canvas-grid legend
# via input_px / patch_size = {128/16=8, 256/16=16, 512/16=32}.
DV3_ANNOTATE_INPUT_PX = (128, 256, 512)
DV3_BASELINE_ALPHA = 0.55
DV3_MARKER_SIZE = 10


def _pct_label(x: float, _pos: int) -> str:
    pct = x * 100
    return f"{pct:.1f}%" if pct < 1 else f"{pct:.0f}%"


def _inline_linestyle_legend(ax: Axes) -> None:
    """Small inside-axes legend disambiguating DINOv3 (dashed) vs CanViT (solid).

    Vertically stacked, upper-left, one size below LEGEND_SIZE so the legend
    doesn't crowd the data region.
    """
    ax.legend(
        handles=linestyle_legend_handles(MODEL_LINESTYLE, {k: k for k in MODEL_LINESTYLE}),
        ncol=1, loc="upper left",
        fontsize=LEGEND_SIZE - 0.5, frameon=True,
        handlelength=1.8, handletextpad=0.25,
        borderpad=0.2, borderaxespad=0.0, labelspacing=0.3,
    )


def _plot_lowess(
    ax: Axes, curve: dict, color,
    *, x_grid: np.ndarray, linestyle: str, scale: float = 100.0,
) -> None:
    smooth = np.array(curve["smooth"]) * scale
    ax.plot(x_grid, smooth, color=color, linewidth=LINE_WIDTH,
            linestyle=linestyle, alpha=LINE_ALPHA)
    lo = np.array(curve["ci_lo"]) * scale
    hi = np.array(curve["ci_hi"]) * scale
    ax.fill_between(x_grid, lo, hi, color=color, alpha=CI_ALPHA)


def _format_area_axis(ax: Axes, ylabel: str) -> None:
    ax.set_xscale("log")
    ax.set_xlim(X_LOG_MIN, 1.0)
    n_decades = int(-np.log10(X_LOG_MIN)) + 1
    ax.set_xticks([10.0 ** -k for k in range(n_decades - 1, -1, -1)])
    ax.xaxis.set_major_formatter(FuncFormatter(_pct_label))
    ax.set_xlabel("GT mask area (% of scene)", fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)


def _panel_passive(ax: Axes, obj: dict) -> None:
    """Panel A: per-class IoU vs mask area. DINOv3 pinned at 128 px (dashed), CanViT t=0 per grid (solid)."""
    x_grid = np.array(obj["x_grid"])
    drawn_dv3 = 0
    for entry in obj["dv3"]:
        output_grid = entry["resolution"] // DV3_PATCH_SIZE
        if output_grid != 8:
            continue
        _plot_lowess(ax, entry, GRID_COLORS.get(output_grid, NEUTRAL),
                     x_grid=x_grid, linestyle=LS_DV3)
        drawn_dv3 += 1
    drawn_cv = 0
    for entry in obj["canvit_t0"]:
        g = entry["canvas_resolution"]
        if g not in GRID_COLORS:
            continue
        _plot_lowess(ax, entry, GRID_COLORS[g], x_grid=x_grid, linestyle=LS_CV)
        drawn_cv += 1
    log.info("panel A: %d DINOv3 curves + %d CanViT t=0 curves", drawn_dv3, drawn_cv)

    _format_area_axis(ax, "IoU")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("Passive (128² px input)", fontsize=LABEL_SIZE, fontweight="bold")
    _inline_linestyle_legend(ax)


def _panel_delta(ax: Axes, obj: dict) -> None:
    """Panel B: Δ IoU (t_late - t_early) per canvas grid (CanViT only)."""
    x_grid = np.array(obj["x_grid"])
    drawn = 0
    for entry in obj["canvit_delta"]:
        g = entry["canvas_resolution"]
        if g not in GRID_COLORS:
            continue
        _plot_lowess(ax, entry, GRID_COLORS[g], x_grid=x_grid, linestyle=LS_CV)
        drawn += 1
    ax.axhline(0, color="0.5", linewidth=0.5, linestyle=":", zorder=0)
    log.info("panel B: %d Δ curves", drawn)

    t0 = obj["plotting"]["t_delta_early"]
    t1 = obj["plotting"]["t_delta_late"]
    _format_area_axis(ax, rf"IoU($t={t1}$) - IoU($t={t0}$)")
    ax.set_title("Active", fontsize=LABEL_SIZE, fontweight="bold")


def _panel_pareto(ax: Axes, ade20k: dict) -> None:
    """Panel C: EG-C2F Pareto frontier, one line per canvas grid. DINOv3 ViT-B
    probe sweep overlaid as a dashed reference with annotated scatter at the
    three powers-of-two inputs (matches the panel A/B canvas-grid legend)."""
    drawn = 0
    for entry in ade20k["policy_curves"]:
        if entry["policy"] != FRONTIER_POLICY:
            continue
        g = entry["canvas_grid"]
        if g not in GRID_COLORS:
            continue
        pts = entry["per_timestep"]
        flops = np.array([p["cum_gflops"] for p in pts]) * 1e9
        mean = np.array([p["mean"] for p in pts]) * 100
        lo = np.array([p["ci_lo"] for p in pts]) * 100
        hi = np.array([p["ci_hi"] for p in pts]) * 100
        ax.plot(flops, mean, color=GRID_COLORS[g], linewidth=LINE_WIDTH,
                linestyle=LS_CV, alpha=LINE_ALPHA, zorder=3)
        if entry.get("n_runs", 0) >= 2:
            ax.fill_between(flops, lo, hi, color=GRID_COLORS[g], alpha=CI_ALPHA, zorder=2)
        drawn += 1
    log.info("panel C: %d Pareto curves", drawn)

    dv3b = sorted(
        (r for r in ade20k.get("probe_table", []) if r["model"] == "DINOv3 ViT-B/16"),
        key=lambda r: r["gflops"],
    )
    if dv3b:
        gf = np.array([r["gflops"] for r in dv3b]) * 1e9
        miou = [r["miou_pct"] for r in dv3b]
        ax.plot(gf, miou, color=NEUTRAL, linewidth=LINE_WIDTH,
                linestyle=LS_DV3, alpha=DV3_BASELINE_ALPHA, zorder=1)
        for r in dv3b:
            dot_color = GRID_COLORS.get(r["output_grid"], NEUTRAL)
            annotated = r["input_px"] in DV3_ANNOTATE_INPUT_PX
            ax.scatter(
                [r["gflops"] * 1e9], [r["miou_pct"]],
                s=DV3_MARKER_SIZE if annotated else DV3_MARKER_SIZE * 0.55,
                color=dot_color, zorder=4,
            )
            if annotated:
                ax.annotate(
                    rf"${r['input_px']}^2$ px",
                    xy=(r["gflops"] * 1e9, r["miou_pct"]),
                    xytext=(5, 0), textcoords="offset points",
                    ha="left", va="center",
                    fontsize=LEGEND_SIZE - 1, color=NEUTRAL,
                )
        log.info("panel C: %d DINOv3 ViT-B probe rows (%d annotated)",
                 len(dv3b), sum(1 for r in dv3b if r["input_px"] in DV3_ANNOTATE_INPUT_PX))

    ax.set_xscale("log")
    if dv3b:
        ax.set_xlim(left=PARETO_FLOPS_MIN, right=float(max(gf)) * PARETO_XLIM_RIGHT_PAD)
    ax.xaxis.set_major_formatter(FuncFormatter(si_flops_label))
    ax.set_xlabel("Cumulative FLOPs", fontsize=LABEL_SIZE)
    ax.set_ylabel("ADE20K mIoU", fontsize=LABEL_SIZE)
    ax.set_ylim(top=50)
    ax.set_yticks([30, 40, 50])
    ax.set_title("Pareto frontier", fontsize=LABEL_SIZE, fontweight="bold")
    _inline_linestyle_legend(ax)


def plot() -> MplFigure:
    obj = load_json(paths.export_json("ade20k_iou_vs_obj"))
    ade20k = load_json(paths.export_json("ade20k_seg"))
    log.info("ade20k_iou_vs_obj: %d DINOv3 curves, %d CanViT t=0, %d CanViT Δ",
             len(obj["dv3"]), len(obj["canvit_t0"]), len(obj["canvit_delta"]))

    fig, axs = three_panel_figure()
    _panel_passive(axs[0], obj)
    _panel_delta(axs[1], obj)
    _panel_pareto(axs[2], ade20k)

    bottom_legend_with_header(
        fig, header="Output patch grid:",
        entries=[(GRID_COLORS[g], rf"${g}^2$") for g in sorted(GRID_COLORS)],
    )
    return fig


figure = Figure(name="resolution_and_mask_size", plot=plot)
