import json
import logging
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from matplotlib.figure import Figure as MplFigure

from canvit_paper_exporter import paths


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Dataset:
    name: str
    compute: Callable[[], dict]


@dataclass(frozen=True)
class Figure:
    name: str
    plot: Callable[[], MplFigure]


@dataclass(frozen=True)
class Diagram:
    """Two-stage producer: Python prep emits PNGs, typst compile composes them
    into the final SVG and PDF. Same `exports/<name>.<ext>` convention as Figure."""
    name: str
    prep: Callable[[], None]
    typst_source: Path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _fmt_elapsed(seconds: float) -> str:
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    if seconds >= 1:
        return f"{seconds:.1f}s"
    return f"{seconds * 1000:.0f}ms"


def run_dataset(ds: Dataset) -> Path:
    out = paths.export_json(ds.name)
    t0 = time.perf_counter()
    payload = ds.compute()
    dt_compute = time.perf_counter() - t0
    t1 = time.perf_counter()
    write_json(out, payload)
    dt_write = time.perf_counter() - t1
    log.info("dataset %s: compute %s + write %s -> %s",
             ds.name, _fmt_elapsed(dt_compute), _fmt_elapsed(dt_write), out)
    return out


FIGURE_FONT = "TeXGyreHeros"


def _check_fonts(pdf: Path) -> None:
    """Every font embedded in the PDF is the figures' font, which ultraplot bundles.

    matplotlib falls back to DejaVu Sans without an error when it cannot find it; this has
    happened in a run that rebuilt matplotlib's font cache, and a rerun restored the font.
    """
    fonts = set(re.findall(rb"/BaseFont /(?:[A-Z]{6}\+)?([\w-]+)", pdf.read_bytes()))
    other = sorted(f.decode() for f in fonts if not f.startswith(FIGURE_FONT.encode()))
    assert fonts and not other, f"{pdf.name}: fonts {other or 'none'} besides {FIGURE_FONT}"


def run_figure(fig: Figure) -> Path:
    svg = paths.export_svg(fig.name)
    pdf = paths.export_pdf(fig.name)
    png = svg.with_suffix(".png")
    svg.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    mpl_fig = fig.plot()
    dt_plot = time.perf_counter() - t0
    t1 = time.perf_counter()
    # `pad_inches=0` explicit because ultraplot's Figure.savefig ignores the
    # rcParam (see style.py for the matching rcParam set, which catches plain
    # matplotlib saves elsewhere). Without this, every save gets a 0.1-inch
    # white border that's wasted under typst/LaTeX.
    # SVG goes to Typst; PDF goes to LaTeX (native matplotlib vector backend).
    # PNG kept as a low-cost preview (200 dpi).
    mpl_fig.savefig(svg, bbox_inches="tight", pad_inches=0, metadata={"Date": None})
    mpl_fig.savefig(pdf, bbox_inches="tight", pad_inches=0, metadata={"Date": None})
    mpl_fig.savefig(png, bbox_inches="tight", pad_inches=0, dpi=200)
    _check_fonts(pdf)
    dt_save = time.perf_counter() - t1
    log.info("figure %s: plot %s + save(svg+pdf+png) %s -> %s",
             fig.name, _fmt_elapsed(dt_plot), _fmt_elapsed(dt_save), svg)
    return svg


def run_diagram(d: Diagram) -> Path:
    svg = paths.export_svg(d.name)
    pdf = paths.export_pdf(d.name)
    svg.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    d.prep()
    dt_prep = time.perf_counter() - t0
    t1 = time.perf_counter()
    # The root is paper/, so wrappers can image() figures under paper/exports/.
    # One typst compile per output: typst infers the format from the extension.
    for out in (svg, pdf):
        subprocess.run(
            ["typst", "compile",
             "--root", str(paths.PAPER_ROOT),
             str(d.typst_source),
             str(out)],
            check=True,
        )
    dt_compose = time.perf_counter() - t1
    log.info("diagram %s: prep %s + typst(svg+pdf) %s -> %s",
             d.name, _fmt_elapsed(dt_prep), _fmt_elapsed(dt_compose), svg)
    return svg
