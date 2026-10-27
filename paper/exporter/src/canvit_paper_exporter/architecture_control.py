"""CanViT-B against an AdaGlimpse-derived baseline, both pretrained by the same DINOv3 distillation with the JAX/Flax NNX
implementation on Cloud TPU, then probed on ADE20K from frozen features under EG-C2F. Inputs: each model's probe
record, the cost model of that implementation (cumulative forward FLOPs under the segmentation readout, parameters
with the pretraining heads), and the chip-hours of the pretraining runs."""

import csv
import math

from canvit_core.backbone import BACKBONES

from canvit_paper_exporter.core import Dataset, load_json
from canvit_paper_exporter.flops.canvit import CANVIT_B, segmentation_glimpse_flops
from canvit_paper_exporter.paths import Inputs

PROBES = {"canvit": "probe_canvit-g128-dv3_s26875_c32_t4.json", "adaglimpse": "probe_bltok2-dv3_s26875_c32_t4.json"}
COST_MODEL = "cost_model_out_eeb49d3.json"
COST_MODEL_KEYS = {"canvit": "cv", "adaglimpse": "ag"}
CHIP_HOURS = "compute_chip_hours_2026-07-27.csv"
RECORDS = Inputs("architecture_control", (*PROBES.values(), COST_MODEL, CHIP_HOURS))
# Conditions both probe records must share; the table compares the two models under them.
SHARED = ("step", "scene_px", "canvas_grid", "glimpse_px", "mask_px", "train_horizon_t", "max_steps", "feature",
          "layer_norm", "eval_policy", "train_policy", "n_timesteps")


def _probe(arm: str) -> dict:
    record = load_json(RECORDS.file(PROBES[arm]))
    assert record["arm"] == arm and not record["eval_only"], record["arm"]
    assert record["history"][-1] == {"step": record["max_steps"], "miou_by_t": record["miou_by_t"],
                                     "miou_final_t": record["final_t_miou"]}, "probe record is not its final step"
    return record


def _run(record: dict) -> str:
    """gs://<bucket>/runs/<run>/ckpt -> <run>"""
    return record["checkpoint_dir"].split("/runs/")[1].split("/")[0]


def _chip_hours(run: str) -> float:
    """A run's chip-hours, summed over the generations it was relaunched as (<run>-g2, <run>-g3, ...)."""
    with open(RECORDS.file(CHIP_HOURS)) as f:
        rows = [r for r in csv.DictReader(f) if r["run_id"] == run or r["run_id"].startswith(f"{run}-g")]
    assert rows and all(r["kind"] == "train" and r["chip_hours"] for r in rows), (run, rows)
    return sum(float(r["chip_hours"]) for r in rows)


def compute() -> dict:
    probes = {arm: _probe(arm) for arm in PROBES}
    shared = {k: probes["canvit"][k] for k in SHARED}
    assert all({k: p[k] for k in SHARED} == shared for p in probes.values()), "the two probe records differ"
    cost = load_json(RECORDS.file(COST_MODEL))
    # The PyTorch count of the same CanViT-B agrees with the JAX cost model's to within one part in a million.
    pytorch_gflops = segmentation_glimpse_flops(CANVIT_B, canvas_grid_size=shared["canvas_grid"]) / 1e9
    assert math.isclose(cost["cv_gf_per_glimpse"], pytorch_gflops, rel_tol=1e-6), (cost["cv_gf_per_glimpse"], pytorch_gflops)
    arms = {}
    for arm, key in COST_MODEL_KEYS.items():
        probe, gflops = probes[arm], cost["gf"][key]
        assert len(gflops) == len(probe["miou_by_t"]) == shared["n_timesteps"]
        arms[arm] = {
            "pretraining_chip_hours": _chip_hours(_run(probe)),
            "params": cost["params"][key], "baseline_tokens_per_side": probe["baseline_tokens_per_side"],
            "patch_px": BACKBONES[CANVIT_B.backbone_name].patch_size if arm == "canvit" else None,
            "per_timestep": [{"t": t, "miou": m, "cum_gflops": g} for t, (m, g) in enumerate(zip(probe["miou_by_t"], gflops, strict=True))],
        }
    return {"shared": shared, "arms": arms}


dataset = Dataset(name="architecture_control", compute=compute, inputs=(RECORDS,))
