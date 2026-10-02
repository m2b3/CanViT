"""Each model's own training compute and best accuracy, for the talk's training-cost slide. Runs in the paper exporter's
environment, from the talk's directory:

    uv run --project ../../../paper/exporter python -u -m experiments.training_cost.export

Training FLOPs come from the rebuttal's accounting (accounting.py beside this file) and, for
AdaptiveNN and AdaGlimpse's ADE20K pipeline, from baselines.py beside this file (its derivation:
sources/baseline-training-compute.md in the talk), under one convention: each system's own training; pretrained weights and teachers it starts
from are counted on no side (DINOv3 for CanViT-B, DeiT III for AdaGlimpse, MAE or SETR for AME). CanViT-B's adaptation
(ImageNet-1k fine-tuning, the ADE20K probe) is counted with canvit_pytorch.flops from the paper's settings.
Accuracies: the paper's macros for CanViT-B, the talk's sota-history.json (each read in its paper) for the others.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import tyro
from canvit_paper_exporter.flops.canvit import ADE20K_NUM_CLASSES, CANVIT_B
from canvit_pytorch.flops import glimpse_flops, linear_flops, segmentation_probe_flops

from experiments.glimpses import GLIMPSE_PX
from experiments.outputs import DECK_DATA, SITE
from experiments.training_cost import accounting, baselines

TALK = Path(__file__).resolve().parents[2]
EF = 1e18
# The paper's ADE20K probe training (Appendix, "Task: ADE20K segmentation"): steps, batch, R-IID glimpses per scene.
PROBE_STEPS, PROBE_BATCH, PROBE_GLIMPSES = 40_000, 16, 10


@dataclass(frozen=True)
class Config:
    out: Path = DECK_DATA / "training-cost.json"


def macro_int(macros: dict[str, str], name: str) -> int:
    return int(str(macros[name]).replace("{,}", ""))


def main(cfg: Config) -> None:
    bwd = accounting.BWD_MULT
    macros = json.loads((SITE / "assets/paper/data_macros.json").read_text())
    history = json.loads((TALK / "sources/sota-history.json").read_text())

    def published(benchmark: str, model: str) -> float:
        [entry] = [e for e in history[benchmark] if e["model"] == model]
        return entry["value"]

    pretraining = accounting.CANVIT_PRETRAINING + accounting.TEACHER_FEATURES

    # ImageNet-1k LP-FT (the paper's macros): every step runs inkFtNGlimpses glimpses at the 32² canvas through CanViT-B
    # and its classifier on the recurrent CLS token, backward through all of them (full BPTT).
    classifier = linear_flops(num_tokens=1, in_dim=CANVIT_B.backbone_spec.embed_dim, out_dim=1000)
    fine_tuning = (macro_int(macros, "inkFtTotalSteps") * macro_int(macros, "inkFtBatchSize")
                   * macro_int(macros, "inkFtNGlimpses")
                   * (glimpse_flops(CANVIT_B, glimpse_size_px=GLIMPSE_PX, canvas_grid_size=32) + classifier)
                   * (1 + bwd))

    # The ADE20K probe at the 64² canvas: CanViT-B frozen, so forward only; the probe's own backward is a linear layer.
    probe_head = segmentation_probe_flops(grid_size=64, embed_dim=CANVIT_B.canvas_dim, num_classes=ADE20K_NUM_CLASSES)
    probe = PROBE_STEPS * PROBE_BATCH * PROBE_GLIMPSES * (
        glimpse_flops(CANVIT_B, glimpse_size_px=GLIMPSE_PX, canvas_grid_size=64) + (1 + bwd) * probe_head)

    adaglimpse_in1k = [accounting.ADAGLIMPSE_CLASSIFICATION + v for v in accounting.ADAGLIMPSE_PRETRAINING.values()]
    ame = list(accounting.AME_SEGMENTATION.values())

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
        "canvit_parts": {"pretraining": rounded(accounting.CANVIT_PRETRAINING),
                         "teacher_features": rounded(accounting.TEACHER_FEATURES),
                         "imagenet_fine_tuning": rounded(fine_tuning), "ade20k_probe": rounded(probe)},
    }
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    cfg.out.write_text(json.dumps(data, indent=1) + "\n")
    print(json.dumps(data["canvit_parts"]), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Config))
