"""ImageNet-1k top-1 of every ablation checkpoint, frozen, its CLS readout fused with the shared DINOv3
probe: tests the CLS pathway where ablation_seg tests the canvas. Files hold top-5 predictions and labels;
top-1 is computed here."""

from canvit_pytorch.hub.repos import DINOV3_VITB16_IN1K_PROBE

from canvit_paper_exporter.ablations import downstream
from canvit_paper_exporter.ablations.registry import BY_SLUG
from canvit_paper_exporter.core import Dataset
from canvit_paper_exporter.results import config, results_in, top1_curve

RESULTS = results_in("in1k_clf_ablations")


def compute() -> dict:
    return downstream.compute(
        RESULTS,
        curve_of=top1_curve,
        recorded_repos=lambda r: (config(r)["classifier"]["pretrained_repo"], config(r)["classifier"]["probe_repo"]),
        expected_repos=lambda slug, scene, grid: (BY_SLUG[slug].ablation.released_repo, DINOV3_VITB16_IN1K_PROBE),
        provenance={"metric": "in1k_top1_accuracy"},
    )


dataset = Dataset(name="ablation_in1k_clf", compute=compute, inputs=(RESULTS,))
