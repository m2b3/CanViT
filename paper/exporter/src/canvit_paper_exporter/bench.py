"""Inference latency and peak memory of CanViT-B and DINOv3 by device, precision, thread count and scene size, from
the JSONL records of canvit_pytorch.evaluate latency-matrix, and the figure that plots them."""

import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import numpy as np
import ultraplot as uplt
from canvit_pytorch.teacher import DINOV3_PATCH_SIZE
from matplotlib.axes import Axes
from matplotlib.figure import Figure as MplFigure

from canvit_paper_exporter.core import Dataset, Figure
from canvit_paper_exporter.paths import Inputs
from canvit_paper_exporter.style import (
    FIGWIDTH, LABEL_SIZE, LEGEND_SIZE, LINE_WIDTH, TICK_SIZE,
    bottom_legend_with_header,
)

log = logging.getLogger(__name__)


RECORDS = Inputs("latency_bench", ("bench_*.jsonl",))
MARKER_SIZE = 4

# Latency samples in the same bin (log10 ms) share one marker, drawn with the opacity their
# overlapping markers composite to; a bin is a fraction of a pixel at 300 dpi.
SCATTER_BIN_DECADES = 0.001

MODEL_STYLES: dict[str, dict] = {
    "canvit":        {"color": "#1f77b4", "label": "CanViT-B",     "marker": "o"},
    "dinov3-vitb16": {"color": "#d62728", "label": "DINOv3 ViT-B", "marker": "s"},
    "dinov3-vits16": {"color": "#9467bd", "label": "DINOv3 ViT-S", "marker": "^"},
}

DTYPE_LS: dict[str, tuple[str, str]] = {
    "amp-bf16": ("-",  "bf16"),
    "fp32":     ("--", "fp32"),
}

DEVICE_LABELS: dict[str, str] = {
    "NVIDIA GeForce RTX 4090":                "RTX 4090",
    "AMD Ryzen 9 7950X 16-Core Processor":   "Ryzen 9 7950X",
}

# CPU linestyle rank — solid for the most-threaded run, dashed for less.
CPU_LS_ORDER = ("-", "--", ":", "-.")


def _load_jsonl(path: Path) -> tuple[dict, list[float], list[float]]:
    """A pass's metadata, the latencies (ms) of its timed iterations, and its peak memory (MB), which CUDA records."""
    meta, *rows = (json.loads(line) for line in path.read_text().strip().split("\n"))
    assert meta["type"] == "meta", path.name
    warm: list[float] = []
    peak_mem: list[float] = []
    for row in rows:
        match row["type"]:
            case "iter":
                warm.append(row["ms"])
            case "peak_mem":
                peak_mem.append(row["peak_mem_mb"])
            case "warmup":
                pass
            case other:
                raise ValueError(f"{path.name}: unknown row type {other!r}")
    assert len(peak_mem) == (1 if meta["device"] == "cuda" else 0), f"{path.name}: peak memory {peak_mem}"
    return meta, warm, peak_mem


def _canvas_grid(meta: dict) -> int | None:
    """CanViT's canvas grid; DINOv3 has none."""
    assert ("canvas_grid" in meta) == (meta["model"] == "canvit"), meta["run_id"]
    return int(meta["canvas_grid"]) if meta["model"] == "canvit" else None


_BENCH_PARAM_KEYS: tuple[str, ...] = ("warmup_iters", "time_budget_s", "max_iters", "min_iters")


def _cfg_key(meta: dict) -> tuple:
    """Identity for merging passes: same hardware + model + shape + threading."""
    return (
        meta["model"], meta["device"], meta["dtype"],
        int(meta["scene_px"]), int(meta["num_threads_actual"]),
        _canvas_grid(meta),
    )


def _compute() -> dict:
    # A profile runs each config in several passes, each writing its own JSONL; a config's row pools the timed
    # iterations of all its passes.
    raw: dict[tuple, dict] = {}
    bench_params: dict[str, set] = {k: set() for k in _BENCH_PARAM_KEYS}
    files = RECORDS.files()
    for f in files:
        meta, warm, peak_mem = _load_jsonl(f)
        assert warm, f"{f.name}: no warm iterations"
        k = _cfg_key(meta)
        if k not in raw:
            raw[k] = {
                "model": meta["model"],
                "device": meta["device"],
                "device_name": meta["device_name"],
                "scene_px": int(meta["scene_px"]),
                "dtype": meta["dtype"],
                "num_threads_actual": int(meta["num_threads_actual"]),
                "warm_ms": [],
                "peak_mem_mb": [],
                "n_passes": 0,
            }
            if meta["model"] == "canvit":
                raw[k]["canvas_grid"] = _canvas_grid(meta)
        raw[k]["warm_ms"].extend(warm)
        raw[k]["peak_mem_mb"].extend(peak_mem)
        raw[k]["n_passes"] += 1
        for pk in _BENCH_PARAM_KEYS:
            bench_params[pk].add(meta[pk])

    configs: list[dict] = []
    for c in raw.values():
        warm = np.array(c["warm_ms"])
        # Peak memory: max across passes; CPU passes record none.
        pm = c.pop("peak_mem_mb")
        c["peak_mem_mb"] = max(pm) if pm else None
        c["min_ms"] = round(float(warm.min()), 3)
        c["median_ms"] = round(float(np.median(warm)), 3)
        c["warm_ms"] = [round(v, 3) for v in c["warm_ms"]]
        configs.append(c)

    configs.sort(key=lambda c: (c["device"], c["num_threads_actual"], c["dtype"],
                                c["model"], c["scene_px"]))

    # The paper states one set of benchmark parameters, so every record must come from the same profile.
    provenance = {}
    for pk, vals in bench_params.items():
        assert len(vals) == 1, (
            f"hw_bench: heterogeneous {pk} across configs: {sorted(vals)}. "
            f"All of {RECORDS} must come from a single profile run."
        )
        (provenance[pk],) = vals
    n_passes = sorted({c["n_passes"] for c in configs})
    assert len(n_passes) == 1, f"hw_bench: heterogeneous n_passes across configs: {n_passes}"
    provenance["n_passes"] = n_passes[0]

    log.info("bench: %d configs (× %d passes) from %d JSONL files; params=%s",
             len(configs), provenance["n_passes"],
             len(files), provenance)
    return {"_provenance": provenance, "configs": configs}


dataset = Dataset(name="hw_bench", compute=_compute, inputs=(RECORDS,))


def _output_grid(c: dict) -> int:
    """Output grid: canvas grid for CanViT, patch grid for DINOv3."""
    return c["canvas_grid"] if c["model"] == "canvit" else c["scene_px"] // DINOV3_PATCH_SIZE


def _series_axis(c: dict) -> str | int:
    """Per-device series: CUDA splits by dtype, CPU splits by thread count."""
    return c["dtype"] if c["device"] == "cuda" else c["num_threads_actual"]


def _linestyle(device: str, series: str | int, cpu_threads: list[int]) -> str:
    if device == "cuda":
        return DTYPE_LS[str(series)][0]
    ranked = sorted(cpu_threads, reverse=True)
    idx = ranked.index(int(series))
    return CPU_LS_ORDER[min(idx, len(CPU_LS_ORDER) - 1)]


def _aggregate(configs: list[dict]) -> dict[tuple[str, str | int, int], np.ndarray]:
    raw: dict[tuple[str, str | int, int], list[float]] = defaultdict(list)
    for c in configs:
        k = (c["model"], _series_axis(c), _output_grid(c))
        raw[k].extend(c["warm_ms"])
    return {k: np.array(v) for k, v in raw.items()}


def _plot_device(ax: Axes, subset: list[dict], device: str, cpu_threads: list[int]) -> None:
    raw = _aggregate(subset)
    series_keys = sorted({(m, s) for (m, s, _) in raw}, key=lambda k: (k[0], str(k[1])))
    for model, series in series_keys:
        style = MODEL_STYLES[model]
        ls = _linestyle(device, series, cpu_threads)
        grids = sorted({g for (m, s, g), _ in raw.items() if m == model and s == series})
        mins = np.array([raw[(model, series, g)].min() for g in grids])
        for g in grids:
            r = raw[(model, series, g)]
            alpha = max(0.02, min(0.35, 5.0 / len(r)))
            bins, counts = np.unique(np.round(np.log10(r) / SCATTER_BIN_DECADES), return_counts=True)
            rgba = np.tile(mcolors.to_rgba(style["color"]), (len(bins), 1))
            rgba[:, 3] = 1 - (1 - alpha) ** counts
            ax.scatter(np.full(len(bins), g), 10 ** (bins * SCATTER_BIN_DECADES), color=rgba, s=4, zorder=1,
                       clip_on=False)
        # The x range ends at the extreme grids, so their markers straddle the frame; unclipped, they show whole
        # without widening the axes (which would move the paper's layout).
        ax.plot(grids, mins, color=style["color"], marker=style["marker"],
                markersize=MARKER_SIZE, linewidth=LINE_WIDTH, linestyle=ls,
                alpha=0.85, zorder=3, clip_on=False)

    # Axis formatting.
    device_grids = sorted({_output_grid(c) for c in subset})
    ax.set_xscale("log", base=2)
    ax.set_xticks(device_grids)
    ax.set_xticklabels([f"{g}×{g}" for g in device_grids])
    ax.set_yscale("log")
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("Output grid (patches)", fontsize=LABEL_SIZE)

    # Per-panel linestyle legend.
    if device == "cuda":
        handles = [
            mlines.Line2D([], [], color="gray", linestyle=ls, linewidth=LINE_WIDTH, label=tag)
            for ls, tag in DTYPE_LS.values()
        ]
    else:
        ranked = sorted(cpu_threads, reverse=True)
        handles = [
            mlines.Line2D([], [], color="gray",
                          linestyle=CPU_LS_ORDER[min(i, len(CPU_LS_ORDER) - 1)],
                          linewidth=LINE_WIDTH,
                          label=f"{t} thread" + ("s" if t != 1 else ""))
            for i, t in enumerate(ranked)
        ]
    ax.legend(handles=handles, fontsize=LEGEND_SIZE, loc="upper left",
              frameon=True, handletextpad=0.4, borderpad=0.3)


def plot(data: dict) -> MplFigure:
    configs = data["configs"]
    devices = sorted({c["device"] for c in configs})
    cpu_threads = sorted({c["num_threads_actual"] for c in configs if c["device"] == "cpu"})

    fig, axs = uplt.subplots(
        ncols=len(devices), figwidth=FIGWIDTH, refaspect=1.4,
        sharex=False, sharey=False, abc=False,
    )
    for ax, device in zip(axs, devices, strict=True):
        subset = [c for c in configs if c["device"] == device]
        _plot_device(ax, subset, device, cpu_threads)
        short = DEVICE_LABELS[subset[0]["device_name"]]
        prefix = "CPU" if device == "cpu" else "GPU"
        ax.set_title(f"{prefix} ({short})", fontsize=LABEL_SIZE, fontweight="bold")
    axs[0].set_ylabel("Min latency (ms)", fontsize=LABEL_SIZE)

    bottom_legend_with_header(
        fig, header="Model:",
        entries=[(s["color"], s["label"]) for s in MODEL_STYLES.values()],
    )
    log.info("hw_latency: %d devices, %d CPU thread counts, %d total configs",
             len(devices), len(cpu_threads), len(configs))
    return fig


figure = Figure(name="hw_latency", plot=plot, datasets=(dataset,))
