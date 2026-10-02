from typing import NamedTuple

import numpy as np
from scipy.stats import bootstrap


class BootstrapResult(NamedTuple):
    mean: float
    ci_lo: float
    ci_hi: float


def bootstrap_mean_ci(
    values: np.ndarray, *, ci: float = 0.95, n_resamples: int = 10_000, seed: int = 42,
) -> BootstrapResult:
    """Bootstrap CI for the mean of a 1-D sample."""
    assert values.ndim == 1, f"expected 1-D, got shape {values.shape}"
    mean = float(values.mean())
    if len(values) < 2:
        return BootstrapResult(mean, mean, mean)
    r = bootstrap(
        (values,), statistic=np.mean, n_resamples=n_resamples,
        confidence_level=ci, rng=np.random.default_rng(seed), method="percentile",
    )
    return BootstrapResult(mean, float(r.confidence_interval.low), float(r.confidence_interval.high))


def bootstrap_per_column(
    curves: np.ndarray, *, ci: float = 0.95, n_resamples: int = 10_000, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column bootstrap CI over a [n_runs, T] array."""
    assert curves.ndim == 2, f"expected 2-D, got shape {curves.shape}"
    n_runs, T = curves.shape
    mean = curves.mean(axis=0)
    if n_runs < 2:
        return mean, mean.copy(), mean.copy()
    lo = np.empty(T)
    hi = np.empty(T)
    for t in range(T):
        r = bootstrap_mean_ci(curves[:, t], ci=ci, n_resamples=n_resamples, seed=seed)
        lo[t] = r.ci_lo
        hi[t] = r.ci_hi
    return mean, lo, hi
