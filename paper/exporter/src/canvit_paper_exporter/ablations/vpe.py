"""ADE20K mIoU under EG-C2F of the ablation baseline and its no-VPE variant at two geometries: scene 512 with canvas 32
(ablation_seg) and scene 1024 with canvas 64 (ade20k_seg_vpe_c64, with probes fitted on that canvas)."""

from canvit_paper_exporter.ablations import seg
from canvit_paper_exporter.ablations.registry import BASELINE, SLUG_BY_REPO
from canvit_paper_exporter.core import Dataset
from canvit_paper_exporter.results import Viewing, config, group_by, load_results, miou_curve, results_in, viewing

POLICY = "entropy_coarse_to_fine"
NO_VPE = "no-vpe"
C64_SCENE_SIZE_PX, C64_CANVAS_GRID = 1024, 64
C64_RESULTS = results_in("ade20k_seg_vpe_c64")


def _geometry(scene_size: int, canvas_grid: int, vpe: list[float], no_vpe: list[float]) -> dict:
    return {
        "scene_size": scene_size, "canvas_grid": canvas_grid,
        "per_timestep": [
            {"t": t, "vpe_miou_pct": a, "no_vpe_miou_pct": b, "delta_pp": a - b}
            for t, (a, b) in enumerate(zip(vpe, no_vpe, strict=True))
        ],
    }


def _canvas_32() -> dict:
    ablations = seg.compute()

    def curve(slug: str) -> list[float]:
        summary = ablations["variants"][slug]["policies"][POLICY]
        assert summary["n_runs"] == 1, summary["n_runs"]
        return [p["mean"] * 100 for p in summary["per_timestep"]]

    provenance = ablations["_provenance"]
    return _geometry(provenance["scene_size"], provenance["canvas_grid"], curve(BASELINE.slug), curve(NO_VPE))


def _canvas_64() -> dict:
    results = load_results(C64_RESULTS).values()
    by_slug = group_by(results, lambda r: SLUG_BY_REPO[config(r)["pretrained_repo"]])
    assert sorted(by_slug) == sorted([BASELINE.slug, NO_VPE]), sorted(by_slug)

    def curve(slug: str) -> list[float]:
        (run,) = by_slug[slug]
        assert viewing(run) == Viewing(POLICY, C64_SCENE_SIZE_PX, C64_CANVAS_GRID), viewing(run)
        repos = (config(run)["pretrained_repo"], config(run)["probe_repo"])
        assert repos == seg.expected_repos(slug, C64_SCENE_SIZE_PX, C64_CANVAS_GRID), repos
        return [m * 100 for m in miou_curve(run)]

    return _geometry(C64_SCENE_SIZE_PX, C64_CANVAS_GRID, curve(BASELINE.slug), curve(NO_VPE))


def compute() -> dict:
    return {"policy": POLICY, "geometries": [_canvas_32(), _canvas_64()]}


dataset = Dataset(name="ablation_vpe_seg", compute=compute, inputs=(*seg.dataset.inputs, C64_RESULTS))
