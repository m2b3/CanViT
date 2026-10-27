import matplotlib as mpl
import matplotlib.lines as mlines
import numpy as np
import ultraplot as uplt
from canvit_pytorch.policies import POLICIES, PolicyName
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
# paper/justfile's check-fonts fails a manuscript build that embeds a Type 3 font.
uplt.rc["pdf.fonttype"] = 42
uplt.rc["ps.fonttype"] = 42


# Each viewing policy's line color and label, its name in the paper (canvit_pytorch.policies), in the legend order of
# the main_results figure.
_POLICY_COLORS: dict[PolicyName, str] = {
    "entropy_coarse_to_fine": "#2ca02c",
    "coarse_to_fine":         "#1f77b4",
    "full_then_random":       "#1a1a1a",
    "random":                 "#ff7f0e",
    "repeated_full_scene":    "#8c564b",
    "fine_to_coarse":         "#d62728",
}
assert set(_POLICY_COLORS) == set(POLICIES), "every viewing policy needs a color"
POLICY_STYLES: dict[str, dict] = {
    policy: {"color": color, "label": POLICIES[policy].paper_name} for policy, color in _POLICY_COLORS.items()
}

POLICY_ORDER = list(POLICY_STYLES)

# The learned viewing policies (learned_policies), by training objective; apart from POLICY_STYLES, whose order is the
# results figure's legend.
LEARNED_POLICY_STYLES: dict[str, dict] = {
    "q_regression":    {"color": "#9467bd", "label": "Q-learning"},
    "policy_gradient": {"color": "#17becf", "label": "REINFORCE"},
}


# Full-text-width 3-panel figure. Both `main_results` and
# `resolution_and_mask_size` use these so the two figures match exactly.
FIGWIDTH = 5.5                # inches (manuscript text width)
REFASPECT = 1.3

# A single-axis figure: wider and shorter than the 3-panel default, so it doesn't waste vertical space beside a table.
SINGLE_PANEL_FIGWIDTH = 4.95  # inches
SINGLE_PANEL_FIGHEIGHT = 2.85

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


def mean_and_ci(ax, x, per_timestep: list[dict], *, n_runs: int, color: str, zorder: float = 2, **line) -> None:
    """A curve of fractions, drawn as percentages, and its CI band one layer below when there are runs to resample."""
    mean, lo, hi = (np.array([p[k] for p in per_timestep]) * 100 for k in ("mean", "ci_lo", "ci_hi"))
    ax.plot(x, mean, color=color, linewidth=LINE_WIDTH, zorder=zorder, **line)
    if n_runs >= 2:
        ax.fill_between(x, lo, hi, color=color, alpha=CI_ALPHA, zorder=zorder - 1)


def add_abc_labels(axs) -> None:
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
    add_abc_labels(axs)
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
    bottom_legend_with_headers(fig, [(header, [(color, label, "-") for color, label in entries])])


def bottom_legend_with_headers(fig: MplFigure, groups: list[tuple[str, list[tuple[str, str, str]]]]) -> None:
    """Bottom-of-figure horizontal legend: for each group, a bold header followed by its swatches.

    `groups` is a list of (header, entries) pairs; each entry is (color, label, linestyle), drawn as its curves are.
    """
    headers = [mlines.Line2D([], [], linewidth=0, markersize=0, label=header) for header, _ in groups]
    handles = []
    for header_handle, (_, entries) in zip(headers, groups, strict=True):
        handles.append(header_handle)
        handles += [
            mlines.Line2D([], [], color=color, linewidth=LINE_WIDTH, linestyle=linestyle, label=label)
            for color, label, linestyle in entries
        ]
    # ultraplot's loc="b" is the bottom; `space` is in em-widths, and a negative one overlaps the x tick labels.
    leg = fig.legend(
        handles=handles,
        loc="b", ncols=len(handles), space=3.25,
        fontsize=LEGEND_SIZE, frame=True, edgecolor="0.8", facecolor="white",
        handlelength=1.5, handletextpad=0.4,
        handler_map={h: _HeaderHandler() for h in headers},
    )
    header_labels = {header for header, _ in groups}
    for text in leg.get_texts():
        if text.get_text() in header_labels:
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
