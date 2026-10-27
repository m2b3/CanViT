"""ImageNet-1k top-1 of the paper's CanViT-B after every glimpse, by policy and canvas grid (canvit_pytorch.evaluate
groups in1k_clf_frozen and in1k_clf_finetuned)."""

import numpy as np
import torch
from canvit_pytorch.hub.repos import (
    DINOV3_VITB16_IN1K_PROBE, FINETUNED_IN1K, FLAGSHIP, RELEASED_CANVAS_GRID_SIZE, RELEASED_SCENE_SIZE_PX,
)

from canvit_paper_exporter.core import Dataset
from canvit_paper_exporter.paths import Inputs
from canvit_paper_exporter.results import Result, config, group_by, load_results, results_in, top1_curve, viewing
from canvit_paper_exporter.stats import per_timestep_ci
from canvit_paper_exporter.style import POLICY_STYLES

BASELINE_RESOLUTION = (RELEASED_SCENE_SIZE_PX, RELEASED_CANVAS_GRID_SIZE)
FROZEN, FINETUNED = results_in("in1k_clf_frozen"), results_in("in1k_clf_finetuned")


def _summarise(runs: list[Result]) -> dict:
    labels = runs[0]["labels"]
    assert all(torch.equal(r["labels"], labels) for r in runs), "runs scored different label sequences"
    per_t = per_timestep_ci(np.array([top1_curve(r) for r in runs]))
    best_t = int(np.argmax([p["mean"] for p in per_t]))
    return {
        "n_runs": len(runs),
        "n_timesteps": len(per_t),
        "per_timestep": per_t,
        "best_t": best_t,
        "best_mean": per_t[best_t]["mean"],
    }


def _compute(runs: Inputs, classifier: dict) -> dict:
    """Every run, which must all evaluate classifier (the recorded config.classifier)."""
    results = load_results(runs).values()
    for r in results:
        assert config(r)["classifier"] == classifier, config(r)["classifier"]
    grouped = group_by(results, viewing)
    is_baseline = {k: (k.scene_size, k.canvas_grid) == BASELINE_RESOLUTION for k in grouped}
    assert any(is_baseline.values()), f"No runs at BASELINE_RESOLUTION={BASELINE_RESOLUTION}"

    configs: list[dict] = []
    sweep: list[dict] = []
    for k, runs in sorted(grouped.items()):
        rec = {
            "policy": k.policy,
            "label": POLICY_STYLES[k.policy]["label"],
            "scene_size": k.scene_size,
            "canvas_grid": k.canvas_grid,
            **_summarise(runs),
        }
        (configs if is_baseline[k] else sweep).append(rec)

    return {
        "_meta": {
            "baseline_scene": BASELINE_RESOLUTION[0],
            "baseline_grid": BASELINE_RESOLUTION[1],
        },
        "configs": configs,
        "canvas_sweep": sweep,
    }


frozen_dataset = Dataset(
    name="in1k_clf_frozen",
    compute=lambda: _compute(FROZEN, {"pretrained_repo": FLAGSHIP, "probe_repo": DINOV3_VITB16_IN1K_PROBE}),
    inputs=(FROZEN,),
)
finetuned_dataset = Dataset(
    name="in1k_clf_finetuned", compute=lambda: _compute(FINETUNED, {"repo": FINETUNED_IN1K}), inputs=(FINETUNED,),
)
