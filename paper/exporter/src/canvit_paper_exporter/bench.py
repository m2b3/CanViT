import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.lines as mlines
import numpy as np
import ultraplot as uplt
from matplotlib.axes import Axes
from matplotlib.figure import Figure as MplFigure

from canvit_paper_exporter import paths
from canvit_paper_exporter.core import Dataset, Figure, load_json
from canvit_paper_exporter.style import (
    FIGWIDTH, LABEL_SIZE, LEGEND_SIZE, LINE_WIDTH, TICK_SIZE,
    bottom_legend_with_header,
)

log = logging.getLogger(__name__)


MARKER_SIZE = 4

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


def _load_jsonl(path: Path) -> tuple[dict, list[float], float | None]:
    lines = path.read_text().strip().split("\n")
    meta = json.loads(lines[0])
    warm: list[float] = []
    peak_mem: float | None = None
    for raw in lines[1:]:
        row = json.loads(raw)
        if row.get("type") == "iter":
            warm.append(row["ms"])
        elif row.get("type") == "peak_mem":
            peak_mem = row.get("peak_mem_mb")
    return meta, warm, peak_mem


_BENCH_PARAM_KEYS: tuple[str, ...] = ("warmup_iters", "time_budget_s", "max_iters", "min_iters")


def _cfg_key(meta: dict) -> tuple:
    """Identity for merging passes: same hardware + model + shape + threading."""
    return (
        meta["model"], meta["device"], meta["dtype"],
        int(meta["scene_px"]), int(meta["num_threads_actual"]),
        meta.get("canvas_grid"),
    )


def _compute() -> dict:
    bench_dir = paths.eval_dir("latency_bench")
    assert bench_dir.is_dir(), f"Missing {bench_dir}"

    # Collect per-jsonl readings keyed by config. `--profile full` runs each config
    # multiple passes (default 3); each pass writes its own jsonl. We pool warm_ms
    # across passes so min/median reflect the full sample, and so the table — which
    # used to over-write per-config rows on later passes — sees a single canonical
    # row per config.
    raw: dict[tuple, dict] = {}
    bench_params: dict[str, set] = {k: set() for k in _BENCH_PARAM_KEYS}
    for f in sorted(bench_dir.glob("bench_*.jsonl")):
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
            if meta.get("canvas_grid") is not None:
                raw[k]["canvas_grid"] = int(meta["canvas_grid"])
        raw[k]["warm_ms"].extend(warm)
        if peak_mem is not None:
            raw[k]["peak_mem_mb"].append(peak_mem)
        raw[k]["n_passes"] += 1
        for pk in _BENCH_PARAM_KEYS:
            if pk in meta:
                bench_params[pk].add(meta[pk])

    configs: list[dict] = []
    for c in raw.values():
        warm = np.array(c["warm_ms"])
        # Peak mem: max across passes (returns None when device is CPU per
        # bench/pt/run.py::_read_peak_mb).
        pm = c.pop("peak_mem_mb")
        c["peak_mem_mb"] = max(pm) if pm else None
        c["min_ms"] = round(float(warm.min()), 3)
        c["median_ms"] = round(float(np.median(warm)), 3)
        c["warm_ms"] = [round(v, 3) for v in c["warm_ms"]]
        configs.append(c)

    configs.sort(key=lambda c: (c["device"], c["num_threads_actual"], c["dtype"],
                                c["model"], c["scene_px"]))

    # Assert single-profile homogeneity. Mixing fast + full results would silently
    # render misleading prose ("warmup_iters = ?"); fail loudly so the operator
    # archives the older pool before exporting.
    for pk, vals in bench_params.items():
        assert len(vals) <= 1, (
            f"hw_bench: heterogeneous {pk} across configs: {sorted(vals)}. "
            f"All bench_*.jsonl in {bench_dir} must come from a single profile run."
        )
    provenance = {pk: vals.pop() for pk, vals in bench_params.items() if vals}
    n_passes = sorted({c["n_passes"] for c in configs})
    assert len(n_passes) == 1, f"hw_bench: heterogeneous n_passes across configs: {n_passes}"
    provenance["n_passes"] = n_passes[0]

    log.info("bench: %d configs (× %d passes) from %d JSONL files; params=%s",
             len(configs), provenance["n_passes"],
             len(list(bench_dir.glob("bench_*.jsonl"))), provenance)
    return {"_provenance": provenance, "configs": configs}


dataset = Dataset(name="hw_bench", compute=_compute)


def _output_grid(c: dict) -> int:
    """Output grid: canvas grid for CanViT, patch grid (scene_px / 16) for DINOv3."""
    return c.get("canvas_grid") or c["scene_px"] // 16


def _series_axis(c: dict) -> str | int:
    """Per-device series: CUDA splits by dtype, CPU splits by thread count."""
    return c["dtype"] if c["device"] == "cuda" else c["num_threads_actual"]


def _linestyle(device: str, series: str | int, cpu_threads: list[int]) -> str:
    if device == "cuda":
        return DTYPE_LS[str(series)][0]
    ranked = sorted(cpu_threads, reverse=True)
    idx = ranked.index(int(series)) if int(series) in ranked else 0
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
            ax.scatter([g] * len(r), r, color=style["color"], alpha=alpha, s=4, zorder=1)
        ax.plot(grids, mins, color=style["color"], marker=style["marker"],
                markersize=MARKER_SIZE, linewidth=LINE_WIDTH, linestyle=ls,
                alpha=0.85, zorder=3)

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


def plot() -> MplFigure:
    data = load_json(paths.export_json("hw_bench"))
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
        device_name = str(subset[0].get("device_name", device))
        short = DEVICE_LABELS.get(device_name, device_name)
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


figure = Figure(name="hw_latency", plot=plot)
