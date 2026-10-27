"""How closely every ablation checkpoint predicts its teacher's features after every glimpse of the ablation study's
episodes (canvit_pytorch.evaluate batch group recon)."""

from dataclasses import asdict

import numpy as np
from canvit_pytorch.evaluate.tasks.reconstruction import ABLATION_EPISODE, METRICS

from canvit_paper_exporter.ablations.registry import BASELINE, BY_SLUG, SLUG_BY_REPO
from canvit_paper_exporter.core import Dataset
from canvit_paper_exporter.results import Result, config, group_by, load_results, results_in
from canvit_paper_exporter.stats import bootstrap_mean_ci

RESULTS = results_in("recon")


def _summarise(slug: str, runs: list[Result]) -> dict:
    """Bootstrap the per-timestep metrics across runs for one variant."""
    per_run = [run["per_timestep"] for run in runs]
    T = len(per_run[0])
    assert all(len(s) == T for s in per_run), f"{slug!r}: inconsistent timestep counts"
    per_t: list[dict] = []
    for t in range(T):
        entry: dict = {"t": t}
        for metric in METRICS:
            r = bootstrap_mean_ci(np.array([run[t][metric] for run in per_run]))
            entry[metric] = r.mean
            entry[f"{metric}_ci_lo"] = r.ci_lo
            entry[f"{metric}_ci_hi"] = r.ci_hi
        per_t.append(entry)
    return {"slug": slug, "n_runs": len(runs), "n_timesteps": T, "per_timestep": per_t}


def _add_baseline_deltas(variants: dict[str, dict]) -> None:
    bl_final = variants[BASELINE.slug]["per_timestep"][-1]
    for rec in variants.values():
        final = rec["per_timestep"][-1]
        rec["deltas"] = {m: (final[m] - bl_final[m]) / bl_final[m] for m in METRICS}


def compute() -> dict:
    results = load_results(RESULTS).values()
    for r in results:
        assert config(r)["episode"] == asdict(ABLATION_EPISODE), config(r)
    grouped = group_by(results, lambda r: SLUG_BY_REPO[config(r)["pretrained_repo"]])
    assert set(grouped) == set(BY_SLUG), f"Variants without runs: {sorted(set(BY_SLUG) - set(grouped))}"
    variants = {slug: _summarise(slug, grouped[slug]) for slug in BY_SLUG}  # the ablation tables' row order
    (n_timesteps,) = {v["n_timesteps"] for v in variants.values()}
    _add_baseline_deltas(variants)
    return {
        "_provenance": {
            "n_variants": len(variants),
            "n_timesteps": n_timesteps,
            "policy": ABLATION_EPISODE.policy,
            "metrics": list(METRICS),
        },
        "variants": variants,
    }


dataset = Dataset(name="ablation_recon", compute=compute, inputs=(RESULTS,))
