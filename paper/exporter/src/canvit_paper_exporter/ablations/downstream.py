"""Downstream evaluations of every ablation checkpoint (canvit_pytorch.evaluate batch groups
ade20k_seg_ablations and in1k_clf_ablations): grouping, completeness, bootstrap curves, deltas."""

from collections.abc import Callable

import numpy as np
from canvit_pytorch.policies import POLICIES

from canvit_paper_exporter.ablations.registry import BASELINE, BY_SLUG, SLUG_BY_REPO
from canvit_paper_exporter.paths import Inputs
from canvit_paper_exporter.results import Result, group_by, load_results, viewing
from canvit_paper_exporter.stats import CONFIDENCE, per_timestep_ci

NamedResult = tuple[str, Result]
CurveOf = Callable[[Result], list[float]]
RecordedRepos = Callable[[Result], tuple[str, str]]


def _stochastic_n_runs(grouped: dict[tuple[str, str], list[NamedResult]]) -> int | None:
    """Every (variant, policy) cell present, one run for deterministic policies and the same count for
    stochastic ones (partial data would shrink CIs); that count, or None for an all-deterministic slate."""
    policies = sorted({policy for _, policy in grouped})
    missing = [(slug, policy) for slug in BY_SLUG for policy in policies if (slug, policy) not in grouped]
    assert not missing, f"Missing (variant, policy) cells: {missing}"
    counts = {}
    for (slug, policy), runs in grouped.items():
        if POLICIES[policy].deterministic:
            assert len(runs) == 1, f"({slug}, {policy}): {len(runs)} runs, expected 1"
        else:
            counts[slug, policy] = len(runs)
    assert len(set(counts.values())) <= 1, f"Non-uniform stochastic run counts: {counts}"
    return next(iter(counts.values()), None)


def _summary(runs: list[NamedResult], *, curve_of: CurveOf, recorded_repos: RecordedRepos, expected: tuple[str, str]) -> dict:
    for name, result in runs:
        assert recorded_repos(result) == expected, f"{name}: evaluated (model, probe) {recorded_repos(result)}, expected {expected}"
    curves = np.array([curve_of(result) for _, result in runs])  # [n_runs, T]
    return {
        "n_runs": len(runs),
        "n_timesteps": curves.shape[1],
        "model_repo": expected[0],
        "probe_repo": expected[1],
        "source_pt": [name for name, _ in runs],
        "per_timestep": per_timestep_ci(curves),
    }


def _add_baseline_deltas(variants: dict[str, dict]) -> None:
    """Relative change from the baseline at the last glimpse, per policy (the convention of ablation_recon)."""
    baseline = variants[BASELINE.slug]["policies"]
    for record in variants.values():
        record["deltas"] = {}
        for policy, summary in record["policies"].items():
            reference = baseline[policy]["per_timestep"][-1]["mean"]
            assert reference != 0, f"baseline {policy}: zero final mean"
            record["deltas"][policy] = (summary["per_timestep"][-1]["mean"] - reference) / reference


def compute(
    runs: Inputs, *, curve_of: CurveOf, recorded_repos: RecordedRepos,
    expected_repos: Callable[[str, int, int], tuple[str, str]], provenance: dict,
) -> dict:
    """recorded_repos(result): the (model, probe) a result evaluated; expected_repos(slug, scene, grid): the (model,
    probe) each variant's results must record."""
    results = load_results(runs).items()
    ((scene, grid),) = {(v.scene_size, v.canvas_grid) for v in (viewing(r) for _, r in results)}
    grouped = group_by(results, lambda named: (SLUG_BY_REPO[recorded_repos(named[1])[0]], viewing(named[1]).policy))
    n_runs_stochastic = _stochastic_n_runs(grouped)
    policies = sorted({policy for _, policy in grouped})
    variants = {
        slug: {
            "slug": slug,
            "policies": {
                policy: _summary(
                    grouped[slug, policy], curve_of=curve_of, recorded_repos=recorded_repos,
                    expected=expected_repos(slug, scene, grid),
                )
                for policy in policies
            },
        }
        for slug in BY_SLUG
    }
    _add_baseline_deltas(variants)
    return {
        "_provenance": {
            "n_variants": len(variants), "scene_size": scene, "canvas_grid": grid, "policies": policies,
            "n_runs_stochastic": n_runs_stochastic, **provenance, "ci": CONFIDENCE,
        },
        "variants": variants,
    }
