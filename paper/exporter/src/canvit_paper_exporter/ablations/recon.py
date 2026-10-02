import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from canvit_paper_exporter import paths
from canvit_paper_exporter.ablations.registry import BASELINE, BY_SLUG
from canvit_paper_exporter.core import Dataset
from canvit_paper_exporter.stats import bootstrap_mean_ci

# recon_{slug}_{YYYYMMDDTHHMMSSZ}_r{run}.pt (canvit_pytorch.evaluate batch group "recon")
_FILENAME = re.compile(r"^recon_(?P<slug>[\w\-]+?)_\d{8}T\d{6}Z_r\d+\.pt$")

METRICS = ("scene_cos_raw", "cls_cos_raw", "scene_cos_norm", "cls_cos_norm")


def _group(recon_dir: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(recon_dir.glob("*.pt")):
        m = _FILENAME.match(path.name)
        assert m is not None, f"Unparseable filename in {recon_dir}: {path.name!r}"
        assert m["slug"] in BY_SLUG, f"Unknown ablation slug in {path.name!r}"
        grouped[m["slug"]].append(path)
    return dict(grouped)


def _summarise(slug: str, files: list[Path]) -> tuple[dict, str]:
    """Bootstrap the per-timestep metrics across runs for one variant; also the policy the runs recorded."""
    runs = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    (policy,) = {run["metadata"]["config"]["episode"]["policy"] for run in runs}
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
    return {"slug": slug, "n_runs": len(files), "n_timesteps": T, "per_timestep": per_t}, policy


def _add_baseline_deltas(variants: dict[str, dict]) -> None:
    bl_final = variants[BASELINE.slug]["per_timestep"][-1]
    for rec in variants.values():
        final = rec["per_timestep"][-1]
        rec["deltas"] = {m: (final[m] - bl_final[m]) / bl_final[m] for m in METRICS}


def compute() -> dict:
    recon_dir = paths.eval_dir("ablation_recon")
    assert recon_dir.is_dir(), f"Missing {recon_dir}"
    grouped = _group(recon_dir)
    assert set(grouped) == set(BY_SLUG), f"Variants without runs: {sorted(set(BY_SLUG) - set(grouped))}"
    summaries = {slug: _summarise(slug, grouped[slug]) for slug in BY_SLUG}  # the ablation tables' row order
    variants = {slug: summary for slug, (summary, _) in summaries.items()}
    (policy,) = {policy for _, policy in summaries.values()}
    (n_timesteps,) = {v["n_timesteps"] for v in variants.values()}
    _add_baseline_deltas(variants)
    return {
        "_provenance": {
            "n_variants": len(variants),
            "n_timesteps": n_timesteps,
            "policy": policy,
            "metrics": list(METRICS),
        },
        "variants": variants,
    }


dataset = Dataset(name="ablation_recon", compute=compute)
