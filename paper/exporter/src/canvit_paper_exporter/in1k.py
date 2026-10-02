import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from canvit_paper_exporter import paths
from canvit_paper_exporter.core import Dataset
from canvit_paper_exporter.stats import bootstrap_mean_ci
from canvit_paper_exporter.style import POLICY_STYLES


# File names of canvit_pytorch.evaluate batch groups "in1k_clf_frozen" and "in1k_clf_finetuned":
#   in1k_{policy}_s{scene}_c{grid}_{YYYYMMDDTHHMMSSZ}_r{run}.pt
_FILENAME = re.compile(r"^in1k_(?P<policy>[a-z_]+)_s(?P<scene>\d+)_c(?P<grid>\d+)_\d{8}T\d{6}Z_r\d+\.pt$")

BASELINE_RESOLUTION: tuple[int, int] = (512, 32)


@dataclass(frozen=True, order=True)
class PolicyKey:
    """Bootstrap-pooling key for IN1k classification runs."""
    policy: str
    scene: int
    grid: int


def _key(path: Path) -> PolicyKey:
    m = _FILENAME.match(path.name)
    assert m, f"Unparseable IN1k filename: {path.name!r}"
    return PolicyKey(m["policy"], int(m["scene"]), int(m["grid"]))


def _summarise(runs: list[dict]) -> dict:
    labels = runs[0]["labels"]
    for i, r in enumerate(runs[1:], 1):
        assert torch.equal(r["labels"], labels), f"Labels mismatch in run {i}"
    preds = torch.stack([r["top_k_preds"] for r in runs])              # [R, N, T, K]
    top1_correct = (preds[:, :, :, 0] == labels[None, :, None]).float()  # [R, N, T]
    per_run_acc = top1_correct.mean(dim=1).numpy()                      # [R, T]
    R, T = per_run_acc.shape

    per_t = []
    for t in range(T):
        r = bootstrap_mean_ci(per_run_acc[:, t])
        per_t.append({
            "t": t,
            "mean": round(float(r.mean), 6),
            "ci_lo": round(float(r.ci_lo), 6),
            "ci_hi": round(float(r.ci_hi), 6),
        })
    best_t = int(np.argmax([p["mean"] for p in per_t]))
    return {
        "n_runs": R,
        "n_timesteps": T,
        "per_timestep": per_t,
        "best_t": best_t,
        "best_mean": per_t[best_t]["mean"],
    }


def _compute_mode(data_dir: Path) -> dict:
    assert data_dir.is_dir(), f"Missing {data_dir}"

    grouped: dict[PolicyKey, list[dict]] = defaultdict(list)
    for path in sorted(data_dir.glob("in1k_*.pt")):
        run = torch.load(path, map_location="cpu", weights_only=False)
        grouped[_key(path)].append(run)

    assert grouped, f"No in1k_*.pt files in {data_dir}"
    assert any(
        (k.scene, k.grid) == BASELINE_RESOLUTION for k in grouped
    ), f"No runs at BASELINE_RESOLUTION={BASELINE_RESOLUTION}"

    configs: list[dict] = []
    sweep: list[dict] = []
    for k, runs in sorted(grouped.items()):
        assert k.policy in POLICY_STYLES, (
            f"Policy {k.policy!r} has no entry in POLICY_STYLES (canvit_paper_exporter.style); "
            f"every IN1k policy must declare a display label there."
        )
        rec = {
            "policy": k.policy,
            "label": POLICY_STYLES[k.policy]["label"],
            "scene_size": k.scene,
            "canvas_grid": k.grid,
            **_summarise(runs),
        }
        (configs if (k.scene, k.grid) == BASELINE_RESOLUTION else sweep).append(rec)

    return {
        "_meta": {
            "baseline_scene": BASELINE_RESOLUTION[0],
            "baseline_grid": BASELINE_RESOLUTION[1],
        },
        "configs": configs,
        "canvas_sweep": sweep,
    }


def compute_frozen() -> dict:
    return _compute_mode(paths.eval_dir("in1k_frozen"))


def compute_finetuned() -> dict:
    return _compute_mode(paths.eval_dir("in1k_finetuned"))


frozen_dataset = Dataset(name="in1k_clf_frozen", compute=compute_frozen)
finetuned_dataset = Dataset(name="in1k_clf_finetuned", compute=compute_finetuned)
