"""Statistics of latency records (JSONL files from `measure`), printed as tables.

Per configuration, over the pooled timed iterations of all its records: median with a bootstrap
95% confidence interval, minimum, tail percentiles and standard deviation. Configurations that
differ only in CPU thread count are compared by their medians' intervals. Each record is checked
for drift: Spearman correlation between iteration index and latency.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

DRIFT_THRESHOLD = 0.3
"""|Spearman ρ| above which a record's latency trends with time."""


class Configuration(NamedTuple):
    model: str
    device: str
    scene_px: int
    dtype: str
    compiled: bool
    threads: int

    def __str__(self) -> str:
        mode = "compiled" if self.compiled else "eager"
        return f"{self.model}/{self.device}/{self.scene_px}px/{self.dtype}/{mode}/threads={self.threads}"


@dataclass(frozen=True)
class Record:
    path: Path
    meta: dict[str, Any]
    iteration_ms: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "Record":
        meta, *rows = (json.loads(line) for line in path.read_text().splitlines())
        assert meta["type"] == "meta", f"{path}: the first row is not the meta row"
        iteration_ms = []
        for row in rows:
            assert row["type"] in ("warmup", "iter", "peak_mem"), f"{path}: unknown row type {row['type']!r}"
            if row["type"] == "iter":
                iteration_ms.append(row["ms"])
        return cls(path=path, meta=meta, iteration_ms=np.array(iteration_ms))

    @property
    def configuration(self) -> Configuration:
        m = self.meta
        return Configuration(m["model"], m["device"], m["scene_px"], m["dtype"], m["compiled"], m["num_threads_actual"])


def bootstrap_ci(
    values: np.ndarray, statistic: Callable[[np.ndarray], float], *, resamples: int = 2000,
) -> tuple[float, float]:
    """95% percentile-bootstrap confidence interval."""
    rng = np.random.default_rng(42)
    estimates = [statistic(values[rng.integers(0, len(values), len(values))]) for _ in range(resamples)]
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def spearman_rho(values: np.ndarray) -> float:
    """Rank correlation between position and value, without tie correction."""
    return float(np.corrcoef(np.arange(len(values)), np.argsort(np.argsort(values)))[0, 1])


@dataclass(frozen=True, kw_only=True)
class LatencySummary:
    """Statistics of latency records: per configuration, across CPU thread counts, and drift within each record."""

    records: tuple[Path, ...]
    """JSONL records written by `latency` or `latency-matrix`."""

    def run(self) -> None:
        records = [Record.load(path) for path in self.records]
        pooled: dict[Configuration, list[Record]] = {}
        for record in records:
            pooled.setdefault(record.configuration, []).append(record)

        medians: dict[Configuration, tuple[float, tuple[float, float]]] = {}
        print(f"{'configuration':<56} {'records':>7} {'iters':>6} {'median':>8} {'median 95% CI':>18} "
              f"{'min':>8} {'p95':>8} {'p99':>8} {'std':>7}")
        for configuration in sorted(pooled):
            ms = np.concatenate([record.iteration_ms for record in pooled[configuration]])
            median, ci = float(np.median(ms)), bootstrap_ci(ms, np.median)
            medians[configuration] = (median, ci)
            print(f"{configuration!s:<56} {len(pooled[configuration]):>7} {len(ms):>6} {median:>8.2f} "
                  f"{f'[{ci[0]:.2f}, {ci[1]:.2f}]':>18} {ms.min():>8.2f} {np.percentile(ms, 95):>8.2f} "
                  f"{np.percentile(ms, 99):>8.2f} {ms.std():>7.2f}")

        print("\nAcross CPU thread counts, same model, scene, precision and compilation:")
        by_threads: dict[Configuration, list[Configuration]] = {}
        for configuration in medians:
            by_threads.setdefault(configuration._replace(threads=0), []).append(configuration)
        for configurations in by_threads.values():
            for a, b in zip(configurations, configurations[1:]):
                (median_a, ci_a), (median_b, ci_b) = medians[a], medians[b]
                overlap = ci_a[1] >= ci_b[0] and ci_b[1] >= ci_a[0]
                change = "CIs overlap" if overlap else f"{100 * (median_b / median_a - 1):+.1f}%"
                print(f"  {a} -> {b.threads} threads: {median_a:.2f} -> {median_b:.2f} ms, {change}")

        print(f"\nDrift within records, |Spearman ρ| > {DRIFT_THRESHOLD}, records under 10 iterations skipped:")
        for record in records:
            if len(record.iteration_ms) >= 10 and abs(rho := spearman_rho(record.iteration_ms)) > DRIFT_THRESHOLD:
                print(f"  {record.path.name}: ρ = {rho:+.3f} ({'slowing' if rho > 0 else 'speeding up'})")
