import torch
from canvit_pytorch import CanViT, CanViTConfig
from canvit_pytorch.hub.repos import RELEASED_CANVAS_GRID_SIZE

from canvit_paper_exporter import paths
from canvit_paper_exporter.ablations.registry import BASELINE, VARIANTS, AblationVariant
from canvit_paper_exporter.core import Dataset, load_json
from canvit_paper_exporter.flops.canvit import segmentation_glimpse_flops

# 4 decimal digits = 0.01% precision; matches the display precision of the ablation table.
REL_DIGITS = 4


def _fractional_change(value: float, baseline: float) -> float:
    """(value - baseline) / baseline, rounded. 0.10 reads as +10% vs baseline."""
    return round((value - baseline) / baseline, REL_DIGITS)


def param_count(config: CanViTConfig) -> int:
    """Parameters of CanViT built from config, pretraining readouts excluded; the meta device allocates nothing."""
    with torch.device("meta"):
        return sum(p.numel() for p in CanViT(config).parameters())


def _gflops(config: CanViTConfig) -> float:
    return segmentation_glimpse_flops(config, canvas_grid_size=RELEASED_CANVAS_GRID_SIZE) / 1e9


def _record(v: AblationVariant, bl_params: int, bl_gflops: float) -> dict:
    p = param_count(v.model_config)
    f = _gflops(v.model_config)
    return {
        "slug": v.slug,
        "label": v.label,
        "delta": v.delta,
        "color": v.color,
        "params": p,
        "params_str": f"{p / 1e6:.1f}M",
        "params_pct": _fractional_change(p, bl_params),
        "gflops_per_glimpse": round(f, REL_DIGITS),
        "gflops_str": f"{f:.1f}",
        "gflops_pct": _fractional_change(f, bl_gflops),
    }


def _baseline_n_steps() -> int:
    """Last training step of the baseline ablation run, from its exported Comet curves."""
    curves = load_json(paths.comet_curves() / f"{BASELINE.slug}.json")
    return max(p["step"] for p in curves["total_loss"])


def compute() -> dict:
    bl_params = param_count(BASELINE.model_config)
    bl_gflops = _gflops(BASELINE.model_config)
    return {
        "_meta": {
            "baseline_slug": BASELINE.slug,
            "order": [BASELINE.slug, *[v.slug for v in VARIANTS]],
            "n_steps": _baseline_n_steps(),
        },
        "variants": {v.slug: _record(v, bl_params, bl_gflops) for v in [BASELINE, *VARIANTS]},
    }


dataset = Dataset(name="ablation_variants", compute=compute)
