"""Downstream evaluations of every ablation checkpoint (canvit_pytorch.evaluate batch groups
ade20k_seg_ablations and in1k_clf_ablations): grouping, completeness, bootstrap curves, deltas."""

import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from canvit_pytorch.policies import POLICIES

from canvit_paper_exporter.ablations.registry import BASELINE, BY_SLUG
from canvit_paper_exporter.stats import bootstrap_per_column

_SLUG_ALTERNATIVES = "|".join(re.escape(s) for s in sorted(BY_SLUG, key=len, reverse=True))
# abl-{slug}_{policy}_s{scene}_c{grid}_{YYYYMMDDTHHMMSSZ}_r{run}.pt
_FILENAME = re.compile(
    rf"^abl-(?P<slug>{_SLUG_ALTERNATIVES})_(?P<policy>[a-z_]+)_s(?P<scene>\d+)_c(?P<grid>\d+)_\d{{8}}T\d{{6}}Z_r\d+\.pt$"
)

Result = dict
CurveOf = Callable[[Result], list[float]]
RecordedRepos = Callable[[Result], tuple[str, str]]


def _group(results_dir: Path) -> tuple[dict[tuple[str, str], list[Path]], int, int]:
    """Files by (slug, policy), and the one (scene, grid) they share."""
    files = sorted(results_dir.glob("*.pt"))
    assert files, f"No .pt files in {results_dir}"
    grouped: dict[tuple[str, str], list[Path]] = defaultdict(list)
    configs: set[tuple[int, int]] = set()
    for path in files:
        m = _FILENAME.match(path.name)
        assert m is not None, f"Unparseable filename in {results_dir}: {path.name!r}"
        grouped[m["slug"], m["policy"]].append(path)
        configs.add((int(m["scene"]), int(m["grid"])))
    assert len(configs) == 1, f"Mixed (scene, grid) configs in {results_dir}: {sorted(configs)}"
    ((scene, grid),) = configs
    return dict(grouped), scene, grid


def _stochastic_n_runs(grouped: dict[tuple[str, str], list[Path]]) -> int | None:
    """Every (variant, policy) cell present, one run for deterministic policies and the same count for
    stochastic ones (partial data would shrink CIs); that count, or None for an all-deterministic slate."""
    policies = sorted({policy for _, policy in grouped})
    missing = [(slug, policy) for slug in BY_SLUG for policy in policies if (slug, policy) not in grouped]
    assert not missing, f"Missing (variant, policy) cells: {missing}"
    counts = {}
    for (slug, policy), files in grouped.items():
        if POLICIES[policy].deterministic:
            assert len(files) == 1, f"({slug}, {policy}): {len(files)} runs, expected 1"
        else:
            counts[slug, policy] = len(files)
    assert len(set(counts.values())) <= 1, f"Non-uniform stochastic run counts: {counts}"
    return next(iter(counts.values()), None)


def _summary(files: list[Path], *, curve_of: CurveOf, recorded_repos: RecordedRepos, expected: tuple[str, str]) -> dict:
    curves = []
    for path in files:
        result = torch.load(path, map_location="cpu", weights_only=False)
        repos = recorded_repos(result)
        assert repos == expected, f"{path.name}: evaluated (model, probe) {repos}, expected {expected}"
        curves.append(curve_of(result))
    arr = np.array(curves)  # [n_runs, T]
    mean, ci_lo, ci_hi = bootstrap_per_column(arr)
    return {
        "n_runs": len(files),
        "n_timesteps": arr.shape[1],
        "model_repo": expected[0],
        "probe_repo": expected[1],
        "source_pt": sorted(f.name for f in files),
        "per_timestep": [
            {"t": t, "mean": round(float(mean[t]), 6), "ci_lo": round(float(ci_lo[t]), 6), "ci_hi": round(float(ci_hi[t]), 6)}
            for t in range(arr.shape[1])
        ],
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
    results_dir: Path, *, curve_of: CurveOf, recorded_repos: RecordedRepos,
    expected_repos: Callable[[str, int, int], tuple[str, str]], provenance: dict,
) -> dict:
    """expected_repos(slug, scene, grid): the (model, probe) each variant's files must record."""
    assert results_dir.is_dir(), f"Missing {results_dir}"
    grouped, scene, grid = _group(results_dir)
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
            "n_runs_stochastic": n_runs_stochastic, **provenance, "ci": 0.95,
        },
        "variants": variants,
    }
