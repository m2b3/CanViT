"""CanViT-B pretrained on ImageNet-1k against the paper's ImageNet-21k checkpoint: frozen ADE20K and ImageNet-1k
results (canvit_pytorch.evaluate groups ade20k_seg_pretrain and in1k_clf_pretrain), and ImageNet-1k results after LP-FT
of each checkpoint (in1k_clf_finetuned_pretrain). Each arm holds its C2F summary and every policy's curve."""

from collections.abc import Callable
from dataclasses import fields
from typing import get_args

import numpy as np
from canvit_pytorch.hub.repos import (
    DINOV3_VITB16_IN1K_PROBE, FINETUNED_IN1K_NNX, PRETRAINED, RELEASED_SCENE_SIZE_PX, PretrainingDataset,
    released_ade20k_probe,
)
from canvit_pytorch.policies import POLICIES
from canvit_pytorch.pretrain.config import PretrainingConfig

from canvit_paper_exporter.core import Dataset
from canvit_paper_exporter.hf import hub_config
from canvit_paper_exporter.paths import Inputs
from canvit_paper_exporter.results import (
    Result, config, group_by, load_results, miou_curve, results_in, top1_curve, viewing,
)
from canvit_paper_exporter.stats import CONFIDENCE, per_timestep_ci

POLICY = "coarse_to_fine"
ADE20K_CANVAS_GRID = 64
IN1K_CANVAS_GRID = 32
ADE20K_FROZEN = results_in("ade20k_seg_pretrain")
IN1K_FROZEN = results_in("in1k_clf_pretrain")
IN1K_LPFT = results_in("in1k_clf_finetuned_pretrain")

Runs = dict[str, list[Result]]


def _runs(inputs: Inputs, canvas_grid: int, model: Callable[[dict], str]) -> Runs:
    """Every run of inputs at canvas_grid, by the model repo that model(config) reads."""
    results = load_results(inputs).values()
    selected = (r for r in results if viewing(r)[1:] == (RELEASED_SCENE_SIZE_PX, canvas_grid))
    return group_by(selected, lambda r: model(config(r)))


def _policies(runs: list[Result], curve: Callable[[Result], list[float]]) -> list[dict]:
    """Every policy's curve; a deterministic policy's runs must be identical and count once."""
    curves = []
    for policy, policy_runs in group_by(runs, lambda r: viewing(r).policy).items():
        values = np.array([curve(r) for r in policy_runs])
        if POLICIES[policy].deterministic:
            assert (values == values[0]).all(), f"{policy}: runs differ"
            values = values[:1]
        else:
            assert len(values) >= 2, f"{len(values)} {policy} runs"
        curves.append({"policy": policy, "n_runs": len(values), "per_timestep": per_timestep_ci(values)})
    return curves


def _arm(runs: list[Result], curve: Callable[[Result], list[float]], **identity: str) -> dict:
    policies = _policies(runs, curve)
    [c2f] = [c for c in policies if c["policy"] == POLICY]
    return {**identity, "n_runs": c2f["n_runs"], "per_timestep": c2f["per_timestep"], "policies": policies}


def _ade20k_frozen(runs: Runs, dataset: PretrainingDataset) -> dict:
    model = PRETRAINED[dataset]
    probe = released_ade20k_probe(dataset, scene_size_px=RELEASED_SCENE_SIZE_PX, canvas_grid_size=ADE20K_CANVAS_GRID)
    assert all(config(r)["probe_repo"] == probe for r in runs[model]), probe
    return _arm(runs[model], miou_curve, model_repo=model, probe_repo=probe)


def _in1k_frozen(runs: Runs, dataset: PretrainingDataset) -> dict:
    model = PRETRAINED[dataset]
    classifier = {"pretrained_repo": model, "probe_repo": DINOV3_VITB16_IN1K_PROBE}
    assert all(config(r)["classifier"] == classifier for r in runs[model]), classifier
    return _arm(runs[model], top1_curve, model_repo=model, probe_repo=DINOV3_VITB16_IN1K_PROBE)


def _in1k_lpft(runs: Runs, dataset: PretrainingDataset) -> dict:
    repo = FINETUNED_IN1K_NNX[dataset]
    assert all(config(r)["classifier"] == {"repo": repo} for r in runs[repo]), repo
    training = hub_config(repo)["training"]
    assert training["base_checkpoint"] == PRETRAINED[dataset], training["base_checkpoint"]
    return {**_arm(runs[repo], top1_curve, model_repo=repo), "training": training}


def compute() -> dict:
    datasets = get_args(PretrainingDataset)
    in1k_pretraining = hub_config(PRETRAINED["in1k"])["metadata"]
    assert in1k_pretraining["dataset"] == "in1k", in1k_pretraining
    ade20k = _runs(ADE20K_FROZEN, ADE20K_CANVAS_GRID, lambda c: c["pretrained_repo"])
    in1k = _runs(IN1K_FROZEN, IN1K_CANVAS_GRID, lambda c: c["classifier"]["pretrained_repo"])
    lpft = _runs(IN1K_LPFT, IN1K_CANVAS_GRID, lambda c: c["classifier"]["repo"])
    return {
        "_provenance": {"policy": POLICY, "scene_size": RELEASED_SCENE_SIZE_PX, "ci": CONFIDENCE},
        "in1k_pretraining_step": in1k_pretraining["step"],
        "pretraining_schedule_steps": next(f.default for f in fields(PretrainingConfig) if f.name == "total_steps"),
        "ade20k_frozen": {"canvas_grid": ADE20K_CANVAS_GRID, "arms": {d: _ade20k_frozen(ade20k, d) for d in datasets}},
        "in1k_frozen": {"canvas_grid": IN1K_CANVAS_GRID, "arms": {d: _in1k_frozen(in1k, d) for d in datasets}},
        "in1k_lpft": {"canvas_grid": IN1K_CANVAS_GRID, "arms": {d: _in1k_lpft(lpft, d) for d in datasets}},
    }


dataset = Dataset(name="dataset_scale", compute=compute, inputs=(ADE20K_FROZEN, IN1K_FROZEN, IN1K_LPFT))
