"""ADE20K mIoU of every ablation checkpoint, with the probe trained on its canvas: the downstream
counterpart of ablation_recon, with the CI protocol of the paper's ADE20K tables."""

from canvit_pytorch.hub.repos import released_ade20k_probe

from canvit_paper_exporter import paths
from canvit_paper_exporter.ablations import downstream
from canvit_paper_exporter.ablations.registry import BY_SLUG
from canvit_paper_exporter.core import Dataset


def _expected_repos(slug: str, scene: int, grid: int) -> tuple[str, str]:
    ablation = BY_SLUG[slug].ablation
    probe = released_ade20k_probe(ablation.probe_model_name, scene_size_px=scene, canvas_grid_size=grid)
    return ablation.released_repo, probe


def compute() -> dict:
    return downstream.compute(
        paths.eval_dir("ablation_seg"),
        curve_of=lambda r: [r["mious"][f"t{t}"] for t in range(len(r["mious"]))],
        recorded_repos=lambda r: (r["metadata"]["config"]["pretrained_repo"], r["metadata"]["config"]["probe_repo"]),
        expected_repos=_expected_repos,
        provenance={},
    )


dataset = Dataset(name="ablation_seg", compute=compute)
