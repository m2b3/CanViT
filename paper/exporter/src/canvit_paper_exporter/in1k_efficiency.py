"""ImageNet-1k top-1 against inference cost at a fixed number of glimpses: CanViT-B under C2F, frozen and after LP-FT,
at canvas 32 and 8, and the released AdaptiveNN and GFNet checkpoints with early exit disabled, each at a single glimpse
and at several (CanViT-B at CANVIT_GLIMPSES, a baseline at every glimpse its released model takes). Top-1 is over the
full validation set. FLOPs count matrix products and convolutions, policy networks and classifiers included, at 2 FLOPs
per multiply-accumulate. Parameters are those of the modules a budget runs, classifiers included: for CanViT-B, the
model and its ImageNet-1k readout of the recurrent CLS token (fine-tuned, or fused with the DINOv3 probe). The
baselines' values come from records of their re-evaluation: top-1 from their released code and checkpoints, FLOPs
counted with torch.utils.flop_counter over their inference loop, and the parameters of the modules their inference
code runs (baselines/gfnet-adaptivenn, active_params)."""

from typing import Any, Literal, NamedTuple

import numpy as np
import torch
from canvit_pytorch import CanViTConfig, CanViTForImageClassification
from canvit_pytorch.benchmarks.imagenet import NUM_CLASSES, NUM_VALIDATION_IMAGES
from canvit_pytorch.hub.repos import (
    DINOV3_VITB16_IN1K_PROBE, FINETUNED_IN1K, FLAGSHIP, RELEASED_GLIMPSE_SIZE_PX, RELEASED_SCENE_SIZE_PX, hub_repo,
)

from canvit_paper_exporter import in1k
from canvit_paper_exporter.core import Dataset, load_json
from canvit_paper_exporter.flops.canvit import CANVIT_B, classification_glimpse_flops
from canvit_paper_exporter.paths import Inputs
from canvit_paper_exporter.results import Result, Viewing, config, load_results, results_in, top1_curve, viewing
from canvit_paper_exporter.stats import CONFIDENCE, per_timestep_ci

POLICY = "coarse_to_fine"
CANVIT_GLIMPSES = (1, 3)
# The flagship fine-tuned at canvas 8 by the JAX/Flax NNX trainer; results of its PyTorch export name it by this repo.
FINETUNED_IN1K_CANVAS_8 = hub_repo("canvitb16-add-vpe-finetune-g128px-s512px-c8-in1k-2026-08-01-nnx")


class Arm(NamedTuple):
    readout: Literal["lpft", "frozen"]
    runs: Inputs
    classifier: dict
    canvas_grid: int


_FROZEN = {"pretrained_repo": FLAGSHIP, "probe_repo": DINOV3_VITB16_IN1K_PROBE}
CANVIT_ARMS = {
    "lpft-c32": Arm("lpft", in1k.FINETUNED, {"repo": FINETUNED_IN1K}, 32),
    "lpft-c8": Arm("lpft", results_in("in1k_clf_finetuned_c8"), {"repo": FINETUNED_IN1K_CANVAS_8}, 8),
    "frozen-c32": Arm("frozen", in1k.FROZEN, _FROZEN, 32),
    "frozen-c8": Arm("frozen", in1k.FROZEN, _FROZEN, 8),
}

GFNET_ACCURACY = "gfnet_reverified_accuracy.json"
GFNET_FLOPS = "gfnet_rollout_flops.json"
GFNET_FACTS = "gfnet_checkpoint_facts.json"
ADAPTIVENN_ACCURACY = "adaptivenn_fixed_budget_in1k_2026-07-31.json"
ADAPTIVENN_FLOPS = "adaptivenn_flops.json"
ACTIVE_PARAMS = "active_params.json"
BASELINE_RECORDS = Inputs(
    "in1k_baselines", (GFNET_ACCURACY, GFNET_FLOPS, GFNET_FACTS, ADAPTIVENN_ACCURACY, ADAPTIVENN_FLOPS, ACTIVE_PARAMS),
)
# Read at every fixation, AdaptiveNN runs its classifier after each one, as GFNet's cascade does.
ADAPTIVENN_ACCOUNTING = "head_every_fixation_gflops"
GFNET_CHECKPOINTS = {
    "gfnet-efficientnet-b3": "gf_effb3_144_T4.pth.tar",
    "gfnet-mobilenet-v3-large": "gf_mbv3l100_128_T3.pth.tar",
}
# GFNet's inference.py --eval_mode 1: anytime prediction, every image takes every pass.
GFNET_ANYTIME_EVAL_MODE = 1


def _budget(glimpses: int, top1: dict, *, cum_gflops: float, params_m: float) -> dict:
    """top1: the mean and interval of a fraction, an interval of zero width for a single evaluation."""
    return {"glimpses": glimpses, "top1": top1["mean"], "ci_lo": top1["ci_lo"], "ci_hi": top1["ci_hi"],
            "cum_gflops": cum_gflops, "params_m": params_m}


def _classifier_params(config: CanViTConfig) -> int:
    """Parameters of the classifier every CanViT-B arm loads: the model, then a LayerNorm and a linear map from the
    recurrent CLS token to the ImageNet-1k classes. The meta device allocates nothing."""
    with torch.device("meta"):
        model = CanViTForImageClassification(canvit_config=config, n_classes=NUM_CLASSES)
    return sum(p.numel() for p in model.parameters())


def _single_evaluation(top1_pct: float) -> dict:
    top1 = round(top1_pct / 100, 6)
    return {"mean": top1, "ci_lo": top1, "ci_hi": top1}


def _canvit(arm: Arm, results: list[Result]) -> dict:
    episode = Viewing(POLICY, RELEASED_SCENE_SIZE_PX, arm.canvas_grid)
    runs = [r for r in results if viewing(r) == episode and config(r)["classifier"] == arm.classifier]
    assert len(runs) >= 2, (arm, len(runs))
    labels = runs[0]["labels"]
    assert len(labels) == NUM_VALIDATION_IMAGES and all(torch.equal(r["labels"], labels) for r in runs), arm
    per_t = per_timestep_ci(np.array([top1_curve(r) for r in runs]))
    gflops = classification_glimpse_flops(CANVIT_B, canvas_grid_size=arm.canvas_grid) / 1e9
    params_m = _classifier_params(CANVIT_B) / 1e6
    return {
        "readout": arm.readout, "classifier": arm.classifier, "canvas_grid": arm.canvas_grid,
        "input_px": RELEASED_GLIMPSE_SIZE_PX, "n_runs": len(runs),
        "budgets": [_budget(g, per_t[g - 1], cum_gflops=g * gflops, params_m=params_m) for g in CANVIT_GLIMPSES],
    }


def _adaptivenn(record: dict, flops: dict, params: dict) -> dict:
    """The glance alone, then the glance and every fixation."""
    accuracy = record["accuracy"]
    assert accuracy["n_images"] == NUM_VALIDATION_IMAGES and not accuracy["early_exit"], accuracy
    assert accuracy["glance_input_size"] == accuracy["focus_patch_size"], accuracy
    assert flops["image_px"] == accuracy["input_size"] and flops["sdpa_math_forced"], flops
    assert round(record["checkpoint"]["params_total"] / 1e6, 3) == params["whole_checkpoint_M"], "different checkpoints"

    def budget(fixations: int) -> dict:
        k = str(fixations)
        return _budget(fixations + 1, _single_evaluation(accuracy["top1_by_k"][k]),
                       cum_gflops=flops["per_k_cumulative"][k][ADAPTIVENN_ACCOUNTING],
                       params_m=params["per_k"][k]["needed_M"])

    return {
        "checkpoint": record["checkpoint"]["source"], "code_commit": record["code"]["commit"],
        "input_px": accuracy["glance_input_size"], "n_runs": 1, "budgets": [budget(0), budget(accuracy["seq_l"])],
    }


def _gfnet(
    checkpoint: str, *, accuracy_record: dict, flops_record: dict, facts: list[dict], params_record: dict,
) -> dict:
    """The glance alone, then every pass the released model takes."""
    accuracy, flops = accuracy_record["models"][checkpoint], flops_record["models"][checkpoint]
    params = params_record[checkpoint]
    [fact] = [f for f in facts if f["file"] == checkpoint]
    identity = (fact["backbone"], fact["patch"], fact["T"])
    assert identity == (accuracy["backbone"], accuracy["patch"], accuracy["T"]), checkpoint
    assert identity == (flops["backbone"], flops["patch_px"], flops["T"]), checkpoint
    assert identity == (params["backbone"], params["patch"], params["T"]), checkpoint
    assert flops["cumulative_macs_theirs"] == fact["cum_macs"], f"{checkpoint}: FLOPs compared with another record"
    assert round(fact["params_total"] / 1e6, 3) == params["whole_checkpoint_M"], f"{checkpoint}: different checkpoints"

    def budget(passes: int, params_m: float) -> dict:
        return _budget(passes, _single_evaluation(accuracy["top1_by_passes"][passes - 1]),
                       cum_gflops=flops["cumulative_gflops_our_unit"][passes - 1], params_m=params_m)

    return {
        "checkpoint": checkpoint, "backbone": fact["backbone"], "input_px": fact["patch"], "n_runs": 1,
        "budgets": [budget(1, params["glance_active_M"]), budget(fact["T"], params["cumulative_at_T_M"])],
    }


def compute() -> dict:
    results = {runs: list(load_results(runs).values()) for runs in {arm.runs for arm in CANVIT_ARMS.values()}}

    def record(name: str) -> Any:
        return load_json(BASELINE_RECORDS.file(name))

    gfnet_accuracy, active_params = record(GFNET_ACCURACY), record(ACTIVE_PARAMS)
    assert gfnet_accuracy["n_images"] == NUM_VALIDATION_IMAGES, gfnet_accuracy["n_images"]
    assert gfnet_accuracy["eval_mode"] == GFNET_ANYTIME_EVAL_MODE, gfnet_accuracy["eval_mode"]
    gfnet_inputs = {"accuracy_record": gfnet_accuracy, "flops_record": record(GFNET_FLOPS),
                    "facts": record(GFNET_FACTS), "params_record": active_params["gfnet"]}
    adaptivenn = _adaptivenn(record(ADAPTIVENN_ACCURACY), record(ADAPTIVENN_FLOPS), active_params["adaptivenn_deit_s"])
    return {
        "_provenance": {"n_images": NUM_VALIDATION_IMAGES, "ci": CONFIDENCE},
        "canvit": {
            "policy": POLICY, "scene_size": RELEASED_SCENE_SIZE_PX,
            "arms": {name: _canvit(arm, results[arm.runs]) for name, arm in CANVIT_ARMS.items()},
        },
        "baselines": {
            "adaptivenn-deit-s": adaptivenn,
            **{name: _gfnet(checkpoint, **gfnet_inputs) for name, checkpoint in GFNET_CHECKPOINTS.items()},
        },
    }


dataset = Dataset(
    name="in1k_efficiency", compute=compute,
    inputs=(*dict.fromkeys(arm.runs for arm in CANVIT_ARMS.values()), BASELINE_RECORDS),
)
