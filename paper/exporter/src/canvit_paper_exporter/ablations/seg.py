"""ADE20K mIoU of every ablation checkpoint, with the probe trained on its canvas: the downstream
counterpart of ablation_recon, with the CI protocol of the paper's ADE20K tables."""

from canvit_pytorch.hub.repos import released_ade20k_probe

from canvit_paper_exporter.ablations import downstream
from canvit_paper_exporter.ablations.registry import BY_SLUG
from canvit_paper_exporter.core import Dataset
from canvit_paper_exporter.results import config, miou_curve, results_in

RESULTS = results_in("ade20k_seg_ablations")


def expected_repos(slug: str, scene: int, grid: int) -> tuple[str, str]:
    """The ablation checkpoint and its released probe for scene and grid."""
    ablation = BY_SLUG[slug].ablation
    probe = released_ade20k_probe(ablation.probe_model_name, scene_size_px=scene, canvas_grid_size=grid)
    return ablation.released_repo, probe


def compute() -> dict:
    return downstream.compute(
        RESULTS,
        curve_of=miou_curve,
        recorded_repos=lambda r: (config(r)["pretrained_repo"], config(r)["probe_repo"]),
        expected_repos=expected_repos,
        provenance={},
    )


dataset = Dataset(name="ablation_seg", compute=compute, inputs=(RESULTS,))
