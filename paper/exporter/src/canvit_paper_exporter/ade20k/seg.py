"""ADE20K mIoU of the paper's CanViT-B after every glimpse by policy and canvas grid, its single full-scene glimpse, and
the passive DINOv3 baselines by input size (canvit_pytorch.evaluate group ade20k_seg); FLOPs are analytic."""

import numpy as np
from canvit_pytorch.flops import dinov3_flops
from canvit_pytorch.hub.repos import (
    DINOV3_PROBE_MODEL_NAMES, PRETRAINED, RELEASED_GLIMPSE_SIZE_PX, released_ade20k_probe, released_dinov3_ade20k_probe,
)
from canvit_pytorch.teacher import DINOV3_NAMES, DINOV3_PATCH_SIZE, DINOV3_REPOS

from canvit_paper_exporter.core import Dataset
from canvit_paper_exporter.flops.adaglimpse import adaglimpse_gflops
from canvit_paper_exporter.flops.ame import ame_gflops
from canvit_paper_exporter.flops.canvit import CANVIT_B, CANVIT_B_NAME, segmentation_glimpse_flops
from canvit_paper_exporter.flops.dinov3 import DINOV3_CONFIGS
from canvit_paper_exporter.results import (
    Result, Viewing, config, group_by, load_results, miou_curve, results_in, viewing,
)
from canvit_paper_exporter.stats import CONFIDENCE, per_timestep_ci

RESULTS = results_in("ade20k_seg")
SINGLE_GLIMPSE_MODEL = f"{CANVIT_B_NAME} (t=0, full scene)"
# Canvas grids without a single-glimpse evaluation, whose single-glimpse row is the best first glimpse of their policy
# curves: the full scene, which the policies that start from it all score alike.
SINGLE_GLIMPSE_FROM_CURVES = {64}

# Published baselines: mIoU from the papers; GFLOPs computed analytically so the
# numbers track any future update to the underlying formulas.
#   AME: Pardyl et al. 2023 (arXiv:2303.06457). Table 3 mIoU.
#   AdaGlimpse: Pardyl et al. 2024 (arXiv:2404.03482). Table 3 mIoU.
BASELINES = [
    {"name": "AME (SETR)", "num_glimpses": 8, "miou_pct": 27.6, "gflops": ame_gflops(8)},
    {"name": "AME (MAE)",  "num_glimpses": 8, "miou_pct": 24.4, "gflops": ame_gflops(8)},
    {"name": "AdaGlimpse", "num_glimpses": 4, "miou_pct": 22.7, "gflops": adaglimpse_gflops(4)},
    {"name": "AdaGlimpse", "num_glimpses": 8, "miou_pct": 25.7, "gflops": adaglimpse_gflops(8)},
]


def flagship_viewing(result: Result) -> Viewing:
    """The run's viewing; the run must evaluate the paper's CanViT-B with the released probe for its scene and canvas."""
    v, c = viewing(result), config(result)
    probe = released_ade20k_probe("in21k", scene_size_px=v.scene_size, canvas_grid_size=v.canvas_grid)
    assert (c["pretrained_repo"], c["probe_repo"]) == (PRETRAINED["in21k"], probe), c
    return v


def _cum_gflops(grid: int, n_timesteps: int) -> list[float]:
    per_glimpse = segmentation_glimpse_flops(CANVIT_B, canvas_grid_size=grid)
    return [(t + 1) * per_glimpse / 1e9 for t in range(n_timesteps)]


def _summarise(key: Viewing, runs: list[Result]) -> dict:
    curves = np.array([miou_curve(r) for r in runs])  # [n_runs, T]
    T = curves.shape[1]
    per_t = [
        {**row, "cum_gflops": gflops}
        for row, gflops in zip(per_timestep_ci(curves), _cum_gflops(key.canvas_grid, T), strict=True)
    ]
    return {
        "policy": key.policy,
        "scene_size": key.scene_size,
        "canvas_grid": key.canvas_grid,
        "n_runs": len(runs),
        "n_timesteps": T,
        "per_timestep": per_t,
    }


def _single_glimpse_row(result: Result) -> dict:
    v = flagship_viewing(result)
    assert v.policy == "coarse_to_fine", v  # whose first glimpse is the full scene
    (miou,) = miou_curve(result)
    return {
        "model": SINGLE_GLIMPSE_MODEL,
        "scene_size": v.scene_size,
        "input_px": config(result)["episode"]["glimpse_size_px"],
        "output_grid": v.canvas_grid,
        "gflops": segmentation_glimpse_flops(CANVIT_B, canvas_grid_size=v.canvas_grid) / 1e9,
        "miou_pct": round(100 * miou, 2),
    }


def _dinov3_row(result: Result) -> dict:
    c, metadata = config(result), result["metadata"]
    variant, size = c["variant"], c["input_size_px"]
    probe = released_dinov3_ade20k_probe(DINOV3_PROBE_MODEL_NAMES[variant], input_size_px=size)
    assert (metadata["dinov3_repo"], metadata["probe_repo"]) == (DINOV3_REPOS[variant], probe), metadata
    model = DINOV3_NAMES[variant]
    (miou,) = miou_curve(result)
    return {
        "model": model,
        "input_px": size,
        "output_grid": size // DINOV3_PATCH_SIZE,
        "gflops": dinov3_flops(DINOV3_CONFIGS[model], input_size_px=size) / 1e9,
        "miou_pct": round(100 * miou, 2),
    }


def _single_glimpse_row_from_curves(claims: dict, grid: int) -> dict:
    t0 = claims["single_glimpse_by_grid"][grid]
    return {
        "model": SINGLE_GLIMPSE_MODEL,
        "scene_size": t0["scene_size"],
        "input_px": RELEASED_GLIMPSE_SIZE_PX,
        "output_grid": grid,
        "gflops": t0["gflops"],
        "miou_pct": t0["miou_pct"],
    }


def _compute_claims(curves: list[dict], best_prior: dict) -> dict:
    """Manuscript-prose numbers, each under a name that says what it binds (e.g. single_glimpse_by_grid[32])."""
    prior_gf = best_prior["gflops"]
    prior_miou = best_prior["miou_pct"] / 100

    beats_prior: list[dict] = []
    for c in curves:
        for p in c["per_timestep"]:
            if p["mean"] >= prior_miou:
                beats_prior.append({
                    "key": f'{c["policy"]}_s{c["scene_size"]}_c{c["canvas_grid"]}',
                    "policy": c["policy"],
                    "scene_size": c["scene_size"],
                    "canvas_grid": c["canvas_grid"],
                    "first_beat_t": p["t"],
                    "first_beat_gflops": p["cum_gflops"],
                    "first_beat_miou_pct": round(p["mean"] * 100, 2),
                    "flop_ratio": prior_gf / p["cum_gflops"],
                })
                break

    best_miou_pct = round(max(p["mean"] for c in curves for p in c["per_timestep"]) * 100, 1)

    grids = sorted({c["canvas_grid"] for c in curves})

    # Per-grid single-glimpse (t=0). At t=0 every policy that starts from the
    # full scene shares the same mIoU, so max-over-policies within a grid picks
    # that common value.
    single_glimpse_by_grid: dict[int, dict] = {}
    for g in grids:
        t0s = [(c, c["per_timestep"][0]) for c in curves if c["canvas_grid"] == g]
        best_c, best_t0 = max(t0s, key=lambda ct: ct[1]["mean"])
        single_glimpse_by_grid[g] = {
            "miou_pct": round(best_t0["mean"] * 100, 2),
            "gflops": best_t0["cum_gflops"],
            "canvas_grid": g,
            "scene_size": best_c["scene_size"],
            "flop_ratio": prior_gf / best_t0["cum_gflops"],
        }

    # Per-grid cheapest prior-beat; a grid may never beat the prior.
    cheapest_beat_by_grid: dict[int, dict] = {}
    for g in grids:
        grid_beats = [b for b in beats_prior if b["canvas_grid"] == g]
        if grid_beats:
            cheapest_beat_by_grid[g] = max(grid_beats, key=lambda b: b["flop_ratio"])

    return {
        "best_prior_name": best_prior["name"],
        "best_prior_miou_pct": best_prior["miou_pct"],
        "best_prior_gflops": prior_gf,
        "best_miou_pct": best_miou_pct,
        "beats_prior": beats_prior,
        "cheapest_beat_by_grid": cheapest_beat_by_grid,
        "single_glimpse_by_grid": single_glimpse_by_grid,
    }


def compute() -> dict:
    results = load_results(RESULTS).values()
    # ADE20kSegmentationDINOv3 records a DINOv3 variant; ADE20kSegmentationCanViT a pretrained CanViT and its episode.
    dinov3 = [r for r in results if "variant" in config(r)]
    canvit = [r for r in results if "variant" not in config(r)]
    single = [r for r in canvit if config(r)["episode"]["num_glimpses"] == 1]
    grouped = group_by((r for r in canvit if config(r)["episode"]["num_glimpses"] > 1), flagship_viewing)
    assert len({viewing(r) for r in single}) == len(single), "two single-glimpse runs of one viewing"
    assert len({(config(r)["variant"], config(r)["input_size_px"]) for r in dinov3}) == len(dinov3), "repeated DINOv3 runs"

    curves = [_summarise(k, runs) for k, runs in sorted(grouped.items())]
    best_prior = max(BASELINES, key=lambda b: b["miou_pct"])
    claims = _compute_claims(curves, best_prior)
    without_single = {c["canvas_grid"] for c in curves} - {viewing(r).canvas_grid for r in single}
    assert without_single == SINGLE_GLIMPSE_FROM_CURVES, without_single
    # Stable sort: the CanViT rows keep their order, single-glimpse runs in file order, then grids from the curves.
    probe_rows = sorted(
        [
            *map(_single_glimpse_row, single),
            *map(_dinov3_row, dinov3),
            *(_single_glimpse_row_from_curves(claims, g) for g in sorted(SINGLE_GLIMPSE_FROM_CURVES)),
        ],
        key=lambda r: (r["model"], r["input_px"]),
    )

    return {
        "_provenance": {
            "n_policy_configs": len(curves),
            "canvas_grids_present": sorted({c["canvas_grid"] for c in curves}),
            "n_probe_rows": len(probe_rows),
            "ci": CONFIDENCE,
        },
        "policy_curves": curves,
        "probe_table": probe_rows,
        "baselines": BASELINES,
        "best_prior": best_prior,
        "claims": claims,
    }


dataset = Dataset(name="ade20k_seg", compute=compute, inputs=(RESULTS,))
