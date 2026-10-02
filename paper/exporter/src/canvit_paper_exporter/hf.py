"""Facts recorded in the config.json of released checkpoints and probes, flattened for the paper's tables.

Configs come from the canvit organization on the Hugging Face Hub, or from the local
directory laid out the same way that $CANVIT_HUB_ROOT names.
"""

import json
from pathlib import Path

from canvit_pytorch.hub.loading import hub_file
from canvit_pytorch.hub.repos import FINETUNED_IN1K, HUB_ROOT, hub_repo

from canvit_paper_exporter.core import Dataset

DV3_PROBE_REPO_TEMPLATE = "dinov3-{model}-lvd1689m-in1k-512x512-linear-clf-probe"
DV3_PROBE_DISPLAY = {
    "vits16": "ViT-S/16",
    "vits16plus": "ViT-S+/16",
    "vitb16": "ViT-B/16",
    "vitl16": "ViT-L/16",
    "vith16plus": "ViT-H+/16",
}
# Published IN-ReAL top-1 from DINOv3 paper Table 14; mirrored in the
# dinov3-in1k-probes README "## Performance" table (official / ours column).
DV3_OFFICIAL_REAL_TOP1 = {
    "vits16": 0.870, "vits16plus": 0.880, "vitb16": 0.893,
    "vitl16": 0.902, "vith16plus": 0.903,
}
# The configs of ViT-S/16, ViT-S+/16 and ViT-B/16 record no loss: their trainer had only softmax
# cross-entropy (the code their Comet runs logged). Later configs record the one they were trained with.
UNRECORDED_OBJECTIVE = "softmax"


def _config(repo: str) -> tuple[dict, str]:
    """A repo's config.json and a description of where it was read."""
    source = f"local CANVIT_HUB_ROOT/{Path(repo).name}" if Path(HUB_ROOT).is_dir() else f"HuggingFace config.json from {repo}"
    return json.loads(hub_file(repo, "config.json").read_text()), source


def _compute_in1k_finetune_config() -> dict:
    config, source = _config(FINETUNED_IN1K)
    params = config["training"]["params"]
    assert params["amp_enabled"] in ("true", "false"), params["amp_enabled"]
    return {
        "_source": source,
        "repo": FINETUNED_IN1K,
        "base_checkpoint": config["training"]["base_checkpoint"],
        # The config does not record the optimizer; the fine-tuning script constructs AdamW
        # (canvit_pytorch.specialize.in1k_tpu.train).
        "optimizer": "AdamW",
        "lr": float(params["lr"]),
        "weight_decay": float(params["weight_decay"]),
        "grad_clip": float(params["grad_clip"]),
        "label_smoothing": float(params["label_smoothing"]),
        "warmup_steps": int(params["warmup_steps"]),
        "epochs": int(params["epochs"]),
        "total_steps": int(params["total_steps"]),
        "batch_size": int(params["batch_size"]),
        "n_glimpses": int(params["n_glimpses"]),
        "chunk_size": int(params["chunk_size"]),
        "full_bptt": int(params["chunk_size"]) >= int(params["n_glimpses"]),
        "glimpse_size": int(params["glimpse_size"]),
        "min_viewpoint_scale": float(params["min_viewpoint_scale"]),
        "amp_enabled": params["amp_enabled"] == "true",
        "n_params": int(params["n_params"]),
    }


def _dv3_probe_row(model_key: str) -> dict:
    repo = hub_repo(DV3_PROBE_REPO_TEMPLATE.format(model=model_key))
    cfg, _ = _config(repo)
    tp = cfg["trial_params"]
    vr = cfg["val_results"]
    assert tp["optimizer"] == "adamw", f"{repo}: {tp['optimizer']}; momentum would apply"
    return {
        "model": DV3_PROBE_DISPLAY[model_key],
        "model_key": model_key,
        "embed_dim": cfg["in_features"],
        "hf_repo": repo,
        "optimizer": tp["optimizer"],
        "batch_size": tp["batch_size"],
        "ref_lr": tp["ref_lr"],
        "peak_lr": tp["peak_lr"],
        "weight_decay": tp["weight_decay"],
        "beta1": tp["beta1"],
        "beta2": tp["beta2"],
        "momentum": None,
        "use_dinov3_init": tp["use_dinov3_init"],
        "objective": tp.get("objective", UNRECORDED_OBJECTIVE),
        "n_train_epochs": tp["n_train_epochs"],
        "outer_epochs": tp["outer_epochs"],
        "total_epochs": tp["total_epochs"],
        "total_steps": tp["total_steps"],
        "warmup_fraction": tp["warmup_fraction"],
        "in1k_top1": round(vr["top1"] * 100, 2),
        "in1k_top5": round(vr["top5"] * 100, 2),
        "real_top1": round(vr["real_top1"] * 100, 2),
        "official_real_top1": round(DV3_OFFICIAL_REAL_TOP1[model_key] * 100, 1),
    }


in1k_finetune_config_dataset = Dataset(name="in1k_finetune_config", compute=_compute_in1k_finetune_config)
dv3_probes_dataset = Dataset(
    name="dinov3_probe_data",
    compute=lambda: {
        "_source": "CanViT-owned DINOv3 probe config.json files",
        "probes": [_dv3_probe_row(m) for m in DV3_PROBE_DISPLAY],
    },
)
