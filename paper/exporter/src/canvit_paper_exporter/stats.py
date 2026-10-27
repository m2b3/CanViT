"""Percentile bootstrap confidence intervals of means over evaluation runs."""

from typing import NamedTuple

import numpy as np
from scipy.stats import bootstrap

CONFIDENCE = 0.95
N_RESAMPLES = 10_000
SEED = 42


class BootstrapResult(NamedTuple):
    mean: float
    ci_lo: float
    ci_hi: float


def bootstrap_mean_ci(values: np.ndarray) -> BootstrapResult:
    """Bootstrap CI for the mean of a 1-D sample; one value (a deterministic policy's run) is its own CI."""
    assert values.ndim == 1, f"expected 1-D, got shape {values.shape}"
    mean = float(values.mean())
    if len(values) < 2:
        return BootstrapResult(mean, mean, mean)
    r = bootstrap(
        (values,), statistic=np.mean, n_resamples=N_RESAMPLES,
        confidence_level=CONFIDENCE, rng=np.random.default_rng(SEED), method="percentile",
    )
    return BootstrapResult(mean, float(r.confidence_interval.low), float(r.confidence_interval.high))


def per_timestep_ci(curves: np.ndarray) -> list[dict]:
    """Mean and bootstrap CI over the runs of a [n_runs, T] array at each timestep, rounded to 6 decimals."""
    assert curves.ndim == 2, f"expected [n_runs, T], got shape {curves.shape}"
    intervals = [bootstrap_mean_ci(column) for column in curves.T]
    return [
        {"t": t, "mean": round(float(mean), 6), "ci_lo": round(r.ci_lo, 6), "ci_hi": round(r.ci_hi, 6)}
        for t, (mean, r) in enumerate(zip(curves.mean(axis=0), intervals, strict=True))
    ]
