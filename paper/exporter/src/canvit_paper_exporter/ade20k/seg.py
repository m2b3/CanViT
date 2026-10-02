import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from canvit_pytorch.flops import dinov3_flops
from canvit_pytorch.hub.repos import RELEASED_GLIMPSE_SIZE_PX

from canvit_paper_exporter import paths
from canvit_paper_exporter.core import Dataset, load_result
from canvit_paper_exporter.flops.adaglimpse import adaglimpse_gflops
from canvit_paper_exporter.flops.ame import ame_gflops
from canvit_paper_exporter.flops.canvit import CANVIT_B, CANVIT_B_NAME, segmentation_glimpse_flops
from canvit_paper_exporter.flops.dinov3 import DINOV3_CONFIGS, DINOV3_PATCH_SIZE
from canvit_paper_exporter.stats import bootstrap_per_column


# File names of canvit_pytorch.evaluate batch group "ade20k_seg":
#   {policy}_s{scene}_c{grid}_{YYYYMMDDTHHMMSSZ}_r{run}.pt
_POLICY_RE = re.compile(
    r"^(?P<policy>.+?)_s(?P<scene>\d+)_c(?P<grid>\d+)_\d{8}T\d{6}Z_r\d+\.pt$"
)
# dv3{b,s}_{resolution}px_{timestamp}.pt
_DV3_PROBE_RE = re.compile(r"^(?P<variant>dv3[bs])_(?P<res>\d+)px_(?P<ts>\d{8}T\d{6}Z)\.pt$")
# canvit_s{scene}_c{grid}_{timestamp}.pt   (single-glimpse probe; no _r{run})
_CANVIT_PROBE_RE = re.compile(r"^canvit_s(?P<scene>\d+)_c(?P<grid>\d+)_(?P<ts>\d{8}T\d{6}Z)\.pt$")

_DV3_MODELS = {"dv3b": "DINOv3 ViT-B/16", "dv3s": "DINOv3 ViT-S/16"}


@dataclass(frozen=True, order=True)
class PolicyKey:
    """Bootstrap-pooling key for ADE20K policy runs."""
    policy: str
    scene: int
    grid: int

    @classmethod
    def from_filename(cls, name: str) -> "PolicyKey | None":
        m = _POLICY_RE.match(name)
        if m is None:
            return None
        return cls(m["policy"], int(m["scene"]), int(m["grid"]))


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


def _cum_gflops(grid: int, n_timesteps: int) -> list[float]:
    per_glimpse = segmentation_glimpse_flops(CANVIT_B, canvas_grid_size=grid)
    return [round((t + 1) * per_glimpse / 1e9, 2) for t in range(n_timesteps)]


def _summarise(key: PolicyKey, files: list[Path]) -> dict:
    curves: list[list[float]] = []
    for f in files:
        r = load_result(f)
        mious = r["mious"]
        T = len([k for k in mious if k.startswith("t")])
        curves.append([mious[f"t{t}"] for t in range(T)])
    arr = np.array(curves)                         # [n_runs, T]
    T = arr.shape[1]
    mean, ci_lo, ci_hi = bootstrap_per_column(arr)
    gflops = _cum_gflops(key.grid, T)
    per_t = [
        {
            "t": t,
            "mean": round(float(mean[t]), 6),
            "ci_lo": round(float(ci_lo[t]), 6),
            "ci_hi": round(float(ci_hi[t]), 6),
            "cum_gflops": gflops[t],
        }
        for t in range(T)
    ]
    return {
        "policy": key.policy,
        "scene_size": key.scene,
        "canvas_grid": key.grid,
        "n_runs": len(files),
        "n_timesteps": T,
        "per_timestep": per_t,
    }


def _probe_table(seg_dir: Path) -> list[dict]:
    """Single-glimpse probe comparison rows (DINOv3 + CanViT at t=0)."""
    # Dedup by (model, config) → keep latest timestamp.
    rows: dict[str, dict] = {}
    for f in sorted(seg_dir.glob("*.pt")):  # sorted → latest ts wins
        mdv3 = _DV3_PROBE_RE.match(f.name)
        mcv = _CANVIT_PROBE_RE.match(f.name)
        if mdv3 is not None:
            label = _DV3_MODELS[mdv3["variant"]]
            res = int(mdv3["res"])
            r = load_result(f)
            rows[f"{label}_{res}px"] = {
                "model": label,
                "input_px": res,
                "output_grid": res // DINOV3_PATCH_SIZE,
                "gflops": round(dinov3_flops(DINOV3_CONFIGS[label], input_size_px=res) / 1e9, 2),
                "miou_pct": round(100 * r["mious"]["t0"], 2),
            }
        elif mcv is not None:
            scene = int(mcv["scene"])
            grid = int(mcv["grid"])
            r = load_result(f)
            rows[f"canvit_s{scene}_c{grid}"] = {
                "model": f"{CANVIT_B_NAME} (t=0, full scene)",
                "scene_size": scene,
                "input_px": RELEASED_GLIMPSE_SIZE_PX,
                "output_grid": grid,
                "gflops": round(segmentation_glimpse_flops(CANVIT_B, canvas_grid_size=grid) / 1e9, 2),
                "miou_pct": round(100 * r["mious"]["t0"], 2),
            }
    return sorted(rows.values(), key=lambda r: (r["model"], r["input_px"]))


def _compute_claims(curves: list[dict], best_prior: dict) -> dict:
    """Manuscript-prose numbers, structured so every binding is explicit at the
    call site (e.g. `claims.single_glimpse_by_grid.at(32)`) rather than a
    scalar that hides its provenance."""
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
                    "flop_ratio": round(prior_gf / p["cum_gflops"], 1),
                })
                break

    best_miou_pct = (
        round(max(p["mean"] for c in curves for p in c["per_timestep"]) * 100, 1)
        if curves else 0.0
    )

    grids = sorted({c["canvas_grid"] for c in curves})

    # Per-grid single-glimpse (t=0). At t=0 every policy that starts from the
    # full scene shares the same mIoU, so max-over-policies within a grid picks
    # that common value.
    single_glimpse_by_grid: dict[int, dict] = {}
    for g in grids:
        t0s = [(c, c["per_timestep"][0]) for c in curves if c["canvas_grid"] == g]
        if not t0s:
            continue
        best_c, best_t0 = max(t0s, key=lambda ct: ct[1]["mean"])
        single_glimpse_by_grid[g] = {
            "miou_pct": round(best_t0["mean"] * 100, 2),
            "gflops": best_t0["cum_gflops"],
            "canvas_grid": g,
            "scene_size": best_c["scene_size"],
            "flop_ratio": round(prior_gf / best_t0["cum_gflops"], 1),
        }

    # Per-grid cheapest prior-beat.
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


def _fill_missing_canvit_probe_rows(probe_rows: list[dict], claims: dict) -> list[dict]:
    """Synthesize CanViT t=0 rows for canvas grids that have policy_curves data
    but no dedicated `canvit_s*_c*.pt` single-glimpse file.

    A single-glimpse t=0 eval is deterministic: every policy starting from the
    full scene shares the same mIoU at t=0. So `single_glimpse_by_grid[g]` IS
    the t=0 value for grid g (computed from policy_curves), with FLOPs from
    `_cum_gflops`. Synthesizing from there is identical to running a dedicated
    single-glimpse eval; it just avoids needing a redundant .pt file.

    This matters for c=64: no (s=512, c=64) single-glimpse .pt was ever
    produced (the dedicated probe was only run at s=1024 c=64), so the c=64
    row's data lives in policy_curves only.
    """
    have_grids = {r["output_grid"] for r in probe_rows if r["model"].startswith("CanViT")}
    for g_str, sg in claims["single_glimpse_by_grid"].items():
        g = int(g_str) if isinstance(g_str, str) else g_str
        if g in have_grids:
            continue
        probe_rows.append({
            "model": f"{CANVIT_B_NAME} (t=0, full scene)",
            "scene_size": sg["scene_size"],
            "input_px": RELEASED_GLIMPSE_SIZE_PX,
            "output_grid": g,
            "gflops": sg["gflops"],
            "miou_pct": sg["miou_pct"],
        })
    return sorted(probe_rows, key=lambda r: (r["model"], r["input_px"]))


def compute() -> dict:
    seg_dir = paths.eval_dir("ade20k_seg")
    assert seg_dir.is_dir(), f"Missing {seg_dir}. Regenerate via the producer in the table in README.md."

    grouped: dict[PolicyKey, list[Path]] = defaultdict(list)
    for path in seg_dir.glob("*.pt"):
        k = PolicyKey.from_filename(path.name)
        if k is not None:
            grouped[k].append(path)

    curves = [_summarise(k, sorted(v)) for k, v in sorted(grouped.items())]
    probe_rows = _probe_table(seg_dir)
    best_prior = max(BASELINES, key=lambda b: b["miou_pct"])
    claims = _compute_claims(curves, best_prior)
    probe_rows = _fill_missing_canvit_probe_rows(probe_rows, claims)

    return {
        "_provenance": {
            "n_policy_configs": len(curves),
            "canvas_grids_present": sorted({c["canvas_grid"] for c in curves}),
            "n_probe_rows": len(probe_rows),
            "ci": 0.95,
        },
        "policy_curves": curves,
        "probe_table": probe_rows,
        "baselines": BASELINES,
        "best_prior": best_prior,
        "claims": claims,
    }


dataset = Dataset(name="ade20k_seg", compute=compute)
