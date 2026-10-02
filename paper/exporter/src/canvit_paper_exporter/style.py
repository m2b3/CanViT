import matplotlib as mpl
import matplotlib.lines as mlines
import ultraplot as uplt
from matplotlib.figure import Figure as MplFigure
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle


# Plain-matplotlib `savefig` calls (notebooks, scripts, anything that doesn't
# go through `run_figure`) get zero padding around the tight-cropped content.
# Default 0.1 inch is a 20-px white border at dpi=200, wasted under typst/LaTeX.
# CAVEAT: ultraplot's Figure.savefig ignores this rcParam (its print_figure
# canvas preprocessor injects pad_inches along the way) — `run_figure` in
# core.py passes `pad_inches=0` explicitly to compensate.
mpl.rcParams["savefig.pad_inches"] = 0

# Pin SVG <defs> path IDs so regen is byte-stable; without this, matplotlib
# emits fresh random IDs (m9995b27f45 etc.) every run, polluting `git diff`.
# Set via `uplt.rc` for the same reason the Type-3-font block below does.
uplt.rc["svg.hashsalt"] = "canvit-paper-exporter"

# Force TrueType (fonttype=42) on the pdf and ps backends. Conference
# submission systems commonly require Type 1 or embedded TrueType fonts and
# reject Type 3, which is matplotlib's PDF default; downstream LaTeX
# \includegraphics inclusions carry the embedded fonts into the final PDF
# (verifiable with `pdffonts <output>.pdf | grep "Type 3"`).
#
# Setting via `uplt.rc` (not `mpl.rcParams`) is required: ultraplot's first
# `uplt.subplots()` call re-applies its own theming and would otherwise reset
# `mpl.rcParams["pdf.fonttype"]` back to 3.
#
# Downstream pipelines should add a `pdffonts` regression guard after compile
# to catch silent regressions here (ultraplot upgrade, rcParam removal, etc.).
uplt.rc["pdf.fonttype"] = 42
uplt.rc["ps.fonttype"] = 42


# Single source of truth for policy display name + plot color.
# Iteration order is the legend order used by the comparison figure and every
# dataset exporter that emits `label` (in1k_clf_*, ade20k_seg).
POLICY_STYLES: dict[str, dict] = {
    "entropy_coarse_to_fine": {"color": "#2ca02c", "label": "EG-C2F"},
    "coarse_to_fine":         {"color": "#1f77b4", "label": "C2F"},
    "full_then_random":       {"color": "#1a1a1a", "label": "F-IID"},
    "random":                 {"color": "#ff7f0e", "label": "R-IID"},
    "repeated_full_scene":    {"color": "#8c564b", "label": "RFS"},
    "fine_to_coarse":         {"color": "#d62728", "label": "F2C"},
}

POLICY_ORDER = list(POLICY_STYLES)


# Full-text-width 3-panel figure. Both `main_results` and
# `resolution_and_mask_size` use these so the two figures match exactly.
FIGWIDTH = 5.5                # inches (manuscript text width)
REFASPECT = 1.3

LABEL_SIZE = 8
TICK_SIZE = 7
LEGEND_SIZE = 6

LINE_WIDTH = 1.5
LINE_ALPHA = 0.6              # transparency lets overlapping lines (DINOv3 dashed + CanViT solid) read
CI_ALPHA = 0.15

# Panel label placement in axes-fraction coordinates.
# x < 0 shifts left of the axes (past the y-axis label); y > 1 shifts above.
ABC_X = -0.06
ABC_Y = 1.05
ABC_SIZE = 9
ABC_WEIGHT = "bold"


def _add_abc_labels(axs) -> None:
    for i, ax in enumerate(axs):
        ax.text(ABC_X, ABC_Y, chr(ord("A") + i),
                transform=ax.transAxes,
                fontsize=ABC_SIZE, fontweight=ABC_WEIGHT,
                ha="right", va="bottom")


def three_panel_figure():
    """3-panel full-width figure. Panel labels live outside the axes (left of the y-axis label)."""
    fig, axs = uplt.subplots(
        ncols=3, figwidth=FIGWIDTH, refaspect=REFASPECT,
        sharex=False, sharey=False,
        abc=False,
    )
    for ax in axs:
        ax.tick_params(labelsize=TICK_SIZE)
        ax.grid(False)
    _add_abc_labels(axs)
    return fig, axs


class _HeaderHandler(HandlerBase):
    """Zero-width swatch for legend entries that should render as bold text only."""
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        r = Rectangle((0, 0), 0, 0, fill=False, edgecolor="none", linewidth=0)
        r.set_transform(trans)
        return [r]


def bottom_legend_with_header(fig: MplFigure, *, header: str, entries: list[tuple[str, str]]) -> None:
    """Bottom-of-figure horizontal legend: one bold header followed by color swatches.

    `entries` is a list of (color, label) pairs.
    """
    header_handle = mlines.Line2D([], [], linewidth=0, markersize=0, label=header)
    swatch_handles = [
        mlines.Line2D([], [], color=color, linewidth=LINE_WIDTH, label=label)
        for color, label in entries
    ]
    # ultraplot-specific: loc="b" is its short form for bottom; ncols is plural.
    # `space` is in em-widths (NOT inches — verified against ultraplot
    # internals/docstring.py:179). `space=0` collapses the default em-width
    # gap; negative values overlap x-tick labels (tested -0.02 on
    # main_results — labels bumped into the legend).
    leg = fig.legend(
        handles=[header_handle, *swatch_handles],
        loc="b", ncols=1 + len(swatch_handles), space=3.25,
        fontsize=LEGEND_SIZE, frame=True, edgecolor="0.8", facecolor="white",
        handlelength=1.5, handletextpad=0.4,
        handler_map={header_handle: _HeaderHandler()},
    )
    for text in leg.get_texts():
        if text.get_text() == header:
            text.set_fontweight("bold")


def inline_top_legend(ax, handles) -> None:
    """Small above-axes legend for linestyle disambiguation."""
    ax.legend(
        handles=handles, ncol=len(handles),
        loc="lower left", mode="expand",
        bbox_to_anchor=(0.0, 1.0, 1.0, 0.0),
        fontsize=LEGEND_SIZE, frameon=True,
        handlelength=2.0, handletextpad=0.2, borderpad=0.2,
        columnspacing=0.5, borderaxespad=0.0,
    )


def linestyle_legend_handles(mapping: dict, labels: dict) -> list:
    """Neutral-gray line handles for linestyle disambiguation legends."""
    return [
        mlines.Line2D([], [], color="gray", linewidth=LINE_WIDTH, linestyle=ls, label=labels[k])
        for k, ls in mapping.items()
    ]


def si_flops_label(x: float, _pos: int = 0) -> str:
    """FuncFormatter-compatible SI suffix for FLOP axes: 1e15 -> 'P', 1e12 -> 'T',
    1e9 -> 'G', 1e6 -> 'M'. Without the T tier, log-scale axes spanning 1e9-1e13
    render as ugly 4-digit '10000G' instead of '10T'."""
    if x >= 1e15:
        return f"{x / 1e15:.0f}P"
    if x >= 1e12:
        return f"{x / 1e12:.0f}T"
    if x >= 1e9:
        return f"{x / 1e9:.0f}G"
    if x >= 1e6:
        return f"{x / 1e6:.0f}M"
    return f"{x:.0f}"
