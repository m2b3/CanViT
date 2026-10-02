"""Each model's own training compute and best accuracy, for the talk's training-cost slide. Runs in the paper exporter's
environment, from the talk's directory:

    uv run --no-sync --project ~/code/CanViT-paper-exporter python -u -m experiments.training_cost.export \
        --rebuttal-flops ~/code/CanViT-Toward-AVFMs/rebuttal/training_flops.py

Training FLOPs come from the rebuttal's accounting (~/code/CanViT-Toward-AVFMs/rebuttal/training_flops.py) and, for
AdaptiveNN and AdaGlimpse's ADE20K pipeline, from baselines.py beside this file (its derivation:
sources/baseline-training-compute.md in the talk), both run here and read from their globals, under one convention: each system's own training; pretrained weights and teachers it starts
from are counted on no side (DINOv3 for CanViT-B, DeiT III for AdaGlimpse, MAE or SETR for AME). CanViT-B's adaptation
(ImageNet-1k fine-tuning, the ADE20K probe) is counted with the same exporter formulas from the paper's settings.
Accuracies: the paper's macros for CanViT-B, the talk's sota-history.json (each read in its paper) for the others.
"""

import json
import runpy
from dataclasses import dataclass, replace
from pathlib import Path

import tyro
from canvit_paper_exporter.flops import primitives as P
from canvit_paper_exporter.flops.arch import CANVIT_B
from canvit_paper_exporter.flops.canvit import canvit_forward_flops_per_glimpse

from experiments.outputs import DECK_DATA, SITE
from experiments.training_cost import baselines

TALK = Path(__file__).resolve().parents[2]
EF = 1e18
# The paper's ADE20K probe training (Appendix, "Task: ADE20K segmentation"): steps, batch, R-IID glimpses per scene.
PROBE_STEPS, PROBE_BATCH, PROBE_GLIMPSES = 40_000, 16, 10


@dataclass(frozen=True)
class Config:
    rebuttal_flops: Path
    """the rebuttal's training_flops.py (CanViT-Toward-AVFMs/rebuttal/)"""
    out: Path = DECK_DATA / "training-cost.json"


def macro_int(macros: dict[str, str], name: str) -> int:
    return int(str(macros[name]).replace("{,}", ""))


def main(cfg: Config) -> None:
    rebuttal = runpy.run_path(str(cfg.rebuttal_flops))
    bwd = rebuttal["BWD_MULT"]
    macros = json.loads((SITE / "assets/paper/data_macros.json").read_text())
    history = json.loads((TALK / "sources/sota-history.json").read_text())

    def published(benchmark: str, model: str) -> float:
        [entry] = [e for e in history[benchmark] if e["model"] == model]
        return entry["value"]

    pretraining = rebuttal["CANVIT_PRETRAIN_FLOPS"] + rebuttal["PRECOMPUTE_FLOPS"]

    # ImageNet-1k LP-FT (the paper's macros): every step runs inkFtNGlimpses glimpses at the 32² canvas through CanViT-B
    # and its classifier on the recurrent CLS token, backward through all of them (full BPTT).
    per_glimpse = canvit_forward_flops_per_glimpse(CANVIT_B)
    classifier = P.linear(1, CANVIT_B.backbone_dim, 1000)
    fine_tuning = (macro_int(macros, "inkFtTotalSteps") * macro_int(macros, "inkFtBatchSize")
                   * macro_int(macros, "inkFtNGlimpses") * (per_glimpse.total - per_glimpse.seg_head + classifier)
                   * (1 + bwd))

    # The ADE20K probe at the 64² canvas: CanViT-B frozen, so forward only; the probe's own backward is a linear layer.
    per_glimpse_64 = canvit_forward_flops_per_glimpse(replace(CANVIT_B, canvas_grid=64))
    probe = PROBE_STEPS * PROBE_BATCH * PROBE_GLIMPSES * (per_glimpse_64.total + bwd * per_glimpse_64.seg_head)

    adaglimpse_in1k = [rebuttal["ADAG_CLF_TASK_FLOPS"] + v for v in rebuttal["adag_pretrain_flops"].values()]
    ame = list(rebuttal["AME_SEG_TASK"].values())

    def rounded(flops: float) -> float:
        return round(flops / EF, 2)

    data = {
        "_about": __doc__.split("\n\n", 2)[2].strip(),
        "unit": "EFLOPs (1e18 floating-point operations, 1 multiply-add = 2)",
        "imagenet": [
            {"model": "CanViT-B", "readout": "frozen, linear probe", "eflops": [rounded(pretraining)] * 2,
             "accuracy": float(macros["inkFrozenBest"])},
            {"model": "CanViT-B", "readout": "fine-tuned", "eflops": [rounded(pretraining + fine_tuning)] * 2,
             "accuracy": float(macros["inkFinetunedBest"])},
            {"model": "AdaGlimpse", "readout": "trained end to end",
             "eflops": [rounded(min(adaglimpse_in1k)), rounded(max(adaglimpse_in1k))],
             "accuracy": published("imagenet_top1", "AdaGlimpse (ViT-B, 14 glimpses of 32 px)"),
             "range": "600 epochs of reconstruction pretraining with 49 to 196 glimpses per image (its code against "
                      "its paper), then 100 of classification"},
            {"model": "AdaptiveNN", "readout": "trained end to end", "eflops": [rounded(baselines.ann_total_code)] * 2,
             "accuracy": published("imagenet_top1", "AdaptiveNN-DeiT-S")},
        ],
        "ade20k": [
            {"model": "CanViT-B", "readout": "frozen, linear probe", "eflops": [rounded(pretraining + probe)] * 2,
             "accuracy": float(macros["adeBestMiou"])},
            {"model": "AME", "readout": "from SETR weights, already trained on ADE20K",
             "eflops": [rounded(min(ame)), rounded(max(ame))],
             "accuracy": published("ade20k_miou", "AME (SETR-initialised ViT-L)"),
             "not_counted": "the ViT-L weights it starts from"},
            {"model": "AdaGlimpse", "readout": "trained end to end",
             "eflops": [rounded(baselines.lo), rounded(baselines.hi)],
             "accuracy": published("ade20k_miou", "AdaGlimpse (ViT-B, 8 glimpses of 48 px)"),
             "range": "pretraining, reconstruction, then segmentation training; 49 to 196 pretraining patches, 8 to "
                      "12 segmentation glimpses"},
        ],
        "canvit_parts": {"pretraining": rounded(rebuttal["CANVIT_PRETRAIN_FLOPS"]),
                         "teacher_features": rounded(rebuttal["PRECOMPUTE_FLOPS"]),
                         "imagenet_fine_tuning": rounded(fine_tuning), "ade20k_probe": rounded(probe)},
    }
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    cfg.out.write_text(json.dumps(data, indent=1) + "\n")
    print(json.dumps(data["canvit_parts"]), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Config))
