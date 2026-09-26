"""Distributional stats for bench/pt/run.py JSONL output.

Groups by (model, device, scene, cg, dtype, compiled, threads, batch); reports
median/p5/p95/p99/std with bootstrap CIs; flags pairwise thread-count
regressions and per-run time drift.
"""

import glob
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import tyro

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("analyze")

@dataclass(frozen=True)
class Run:
    path: Path
    meta: dict
    iter_ms: np.ndarray
    iter_wall_s: np.ndarray
    warmup_ms: np.ndarray
    peak_mem_mb: float | None

    @property
    def key(self) -> tuple:
        """Group key: everything except run_id and wall timestamps."""
        m = self.meta
        return (
            m["model"],
            m["device"],
            m["scene_px"],
            m.get("canvas_grid"),
            m["dtype"],
            m["compiled"],
            m["num_threads_actual"],
            m["batch_size"],
        )

    @property
    def label(self) -> str:
        m = self.meta
        cg = f" cg={m['canvas_grid']}" if m.get("canvas_grid") is not None else ""
        c = "compiled" if m["compiled"] else "eager"
        return (
            f"{m['model']}/{m['device']}/s={m['scene_px']}{cg}/"
            f"{m['dtype']}/{c}/threads={m['num_threads_actual']}"
        )


def load_jsonl(path: Path) -> Run:
    lines = path.read_text().strip().split("\n")
    assert lines, f"{path.name}: empty file"
    meta = json.loads(lines[0])
    assert meta.get("type") == "meta", f"{path.name}: first line is not type=meta"

    iter_ms: list[float] = []
    iter_wall: list[float] = []
    warm_ms: list[float] = []
    peak_mb: float | None = None

    for line in lines[1:]:
        row = json.loads(line)
        t = row.get("type")
        if t == "iter":
            iter_ms.append(row["ms"])
            iter_wall.append(row["wall_s"])
        elif t == "warmup":
            warm_ms.append(row["ms"])
        elif t == "peak_mem":
            peak_mb = row["peak_mem_mb"]
        else:
            raise ValueError(f"{path.name}: unknown row type {t!r}")

    return Run(
        path=path,
        meta=meta,
        iter_ms=np.array(iter_ms),
        iter_wall_s=np.array(iter_wall),
        warmup_ms=np.array(warm_ms),
        peak_mem_mb=peak_mb,
    )


def bootstrap_ci(
    arr: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(arr)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = stat_fn(arr[idx])
    return (
        float(np.percentile(boots, 100 * alpha / 2)),
        float(np.percentile(boots, 100 * (1 - alpha / 2))),
    )


@dataclass
class GroupStats:
    key: tuple
    label: str
    runs: list[Run] = field(default_factory=list)

    @property
    def all_ms(self) -> np.ndarray:
        return np.concatenate([r.iter_ms for r in self.runs])

    @property
    def n_iters(self) -> int:
        return int(self.all_ms.size)

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    def summary(self) -> dict:
        a = self.all_ms
        if a.size == 0:
            nan = float("nan")
            return {
                "n_iters": 0, "n_runs": self.n_runs,
                "median": nan, "median_ci": (nan, nan),
                "min": nan, "min_ci": (nan, nan),
                "p5": nan, "p95": nan, "p99": nan,
                "mean": nan, "std": nan, "max": nan,
            }
        med = float(np.median(a))
        med_lo, med_hi = bootstrap_ci(a, np.median)
        mn = float(a.min())
        mn_lo, mn_hi = bootstrap_ci(a, np.min)
        return {
            "n_iters": self.n_iters,
            "n_runs": self.n_runs,
            "median": med,
            "median_ci": (med_lo, med_hi),
            "min": mn,
            "min_ci": (mn_lo, mn_hi),
            "p5": float(np.percentile(a, 5)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "mean": float(a.mean()),
            "std": float(a.std()),
            "max": float(a.max()),
        }


def group_runs(runs: Iterable[Run]) -> dict[tuple, GroupStats]:
    groups: dict[tuple, GroupStats] = {}
    for r in runs:
        g = groups.setdefault(r.key, GroupStats(key=r.key, label=r.label))
        g.runs.append(r)
    return groups


def time_drift_stats(run: Run) -> dict:
    """Spearman rho of iter latency vs iter index. Positive = slows over time."""
    n = int(run.iter_ms.size)
    if n < 10:
        return {"n": n, "spearman_rho": None}
    ranks = np.argsort(np.argsort(run.iter_ms))
    rho = float(np.corrcoef(np.arange(n), ranks)[0, 1])
    return {"n": n, "spearman_rho": rho}


def ci_disjoint(ci_a: tuple[float, float], ci_b: tuple[float, float]) -> bool:
    return ci_a[1] < ci_b[0] or ci_b[1] < ci_a[0]


def print_summary_table(groups: dict[tuple, GroupStats]) -> None:
    if not groups:
        log.info("No groups to report.")
        return

    log.info("")
    log.info(
        f"{'group':<60s} {'n_runs':>6s} {'n_iters':>7s}  "
        f"{'median':>8s}  {'median_ci':>20s}  {'min':>8s}  {'p95':>8s}  {'p99':>8s}  {'std':>7s}"
    )
    log.info("-" * 150)
    for g in sorted(groups.values(), key=lambda x: x.key):
        s = g.summary()
        ci = f"[{s['median_ci'][0]:>6.2f}, {s['median_ci'][1]:>6.2f}]"
        log.info(
            f"{g.label:<60s} {s['n_runs']:>6d} {s['n_iters']:>7d}  "
            f"{s['median']:>8.2f}  {ci:>20s}  {s['min']:>8.2f}  "
            f"{s['p95']:>8.2f}  {s['p99']:>8.2f}  {s['std']:>7.2f}"
        )


def print_pairwise(groups: dict[tuple, GroupStats]) -> None:
    """For groups that differ only in num_threads_actual, check for real regressions."""
    # Bucket by all-except-thread-count (index 6 in the key tuple).
    thread_groups: dict[tuple, list[GroupStats]] = {}
    for g in groups.values():
        sig = g.key[:6] + g.key[7:]
        thread_groups.setdefault(sig, []).append(g)

    log.info("")
    log.info("Pairwise across threads (same model/scene/cg/dtype/compiled/batch):")
    log.info(
        f"{'config':<55s}  {'threads A→B':>14s}  {'med_A':>8s}  {'med_B':>8s}  "
        f"{'CI_overlap':>10s}  {'verdict':<35s}"
    )
    log.info("-" * 150)

    for sig, gs in sorted(thread_groups.items()):
        if len(gs) < 2:
            continue
        gs_sorted = sorted(gs, key=lambda g: g.key[6])
        for a, b in zip(gs_sorted, gs_sorted[1:]):
            sa, sb = a.summary(), b.summary()
            overlap = not ci_disjoint(sa["median_ci"], sb["median_ci"])
            med_direction = np.sign(sb["median"] - sa["median"])
            min_direction = np.sign(sb["min"] - sa["min"])
            agrees = med_direction == min_direction
            verdict = ""
            if med_direction > 0 and not overlap:
                verdict = f"⬆ REAL slowdown (+{(sb['median']/sa['median']-1)*100:.1f}%)"
            elif med_direction < 0 and not overlap:
                verdict = f"⬇ real speedup ({(sb['median']/sa['median']-1)*100:+.1f}%)"
            elif overlap:
                verdict = "~ CIs overlap (noisy)"
            if not agrees:
                verdict += " [med/min disagree]"

            label = a.label.rsplit("/threads=", 1)[0]
            log.info(
                f"{label:<55s}  {a.key[6]:>6d}→{b.key[6]:<7d}  "
                f"{sa['median']:>8.2f}  {sb['median']:>8.2f}  "
                f"{'yes' if overlap else 'no':>10s}  {verdict:<35s}"
            )


def print_drift_flags(runs: list[Run]) -> None:
    log.info("")
    log.info("Per-run time-drift detection (Spearman rho of iter_ms vs iter_index; flag if |rho| > 0.3):")
    log.info(f"{'run':<80s}  {'n':>5s}  {'rho':>8s}  {'flag':<25s}")
    log.info("-" * 125)
    for r in runs:
        s = time_drift_stats(r)
        if s["spearman_rho"] is None:
            log.info(f"{r.path.name:<80s}  {s['n']:>5d}  {'—':>8s}  n<10 skipped")
            continue
        rho = s["spearman_rho"]
        flag = ""
        if abs(rho) > 0.3:
            flag = "DRIFT (slowing)" if rho > 0 else "DRIFT (speeding)"
        log.info(f"{r.path.name:<80s}  {s['n']:>5d}  {rho:>+8.3f}  {flag:<25s}")


@dataclass
class Args:
    pattern: str | None = None
    """Glob pattern for JSONL files (e.g. 'results/*20260417*.jsonl')."""
    files: list[Path] = field(default_factory=list)
    """Explicit JSONL file paths (alternative to --pattern)."""


def main(args: Args) -> None:
    paths: list[Path] = list(args.files)
    if args.pattern:
        paths.extend(sorted(Path(p) for p in glob.glob(args.pattern)))
    assert paths, "Pass --pattern or --files. Nothing to analyze."

    log.info(f"Loading {len(paths)} file(s)...")
    runs = [load_jsonl(p) for p in paths]

    groups = group_runs(runs)
    log.info(f"Loaded {len(runs)} runs in {len(groups)} group(s).")

    print_summary_table(groups)
    print_pairwise(groups)
    print_drift_flags(runs)


if __name__ == "__main__":
    main(tyro.cli(Args))
