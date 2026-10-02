"""ImageNet-1k top-1 of every ablation checkpoint, frozen, its CLS readout fused with the shared DINOv3
probe: tests the CLS pathway where ablation_seg tests the canvas. Files hold top-5 predictions and labels;
top-1 is computed here."""

from canvit_pytorch.hub.repos import DINOV3_VITB16_IN1K_PROBE

from canvit_paper_exporter import paths
from canvit_paper_exporter.ablations import downstream
from canvit_paper_exporter.ablations.registry import BY_SLUG
from canvit_paper_exporter.core import Dataset


def _top1_curve(result: dict) -> list[float]:
    top1 = result["top_k_preds"][:, :, 0]
    return [float(x) for x in (top1 == result["labels"][:, None]).float().mean(dim=0)]


def compute() -> dict:
    return downstream.compute(
        paths.eval_dir("ablation_in1k_clf"),
        curve_of=_top1_curve,
        recorded_repos=lambda r: (
            r["metadata"]["config"]["classifier"]["pretrained_repo"], r["metadata"]["config"]["classifier"]["probe_repo"],
        ),
        expected_repos=lambda slug, scene, grid: (BY_SLUG[slug].ablation.released_repo, DINOV3_VITB16_IN1K_PROBE),
        provenance={"metric": "in1k_top1_accuracy"},
    )


dataset = Dataset(name="ablation_in1k_clf", compute=compute)
