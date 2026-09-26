"""Help for code and checkpoints written for canvit-pytorch 0.1.

canvit-pytorch 0.2 renamed public classes, checkpoint config keys and weight
names. The package calls into this module only after a lookup or a load has
failed, to explain what changed. `convert_checkpoint` rewrites a 0.1 checkpoint
directory or Hub repo in the 0.2 format:

    python -m canvit_pytorch.legacy SOURCE OUT_DIR
"""

import json
import re
from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from torch import Tensor

from canvit_pytorch.hub.repos import OLD_FORMAT_REVISION

UPGRADE_HINT = (
    "canvit-pytorch 0.2 changed its API and checkpoint format, and the released checkpoints on the Hub are in "
    f"the new format. Code written for 0.1 needs `canvit-pytorch<0.2` and the checkpoints' revision "
    f"`{OLD_FORMAT_REVISION}`. A 0.1 checkpoint converts with `python -m canvit_pytorch.legacy SOURCE OUT_DIR`."
)

RENAMED: dict[str, str] = {
    "CanViTForPretrainingHFHub": "CanViTForPretraining",
    "CanViTForPretrainingConfig": "CanViTForPretraining(canvit_config=CanViTConfig(...), teacher_dim=..., teacher_patch_grid=...)",
    "VPEEncoder": "canvit_pytorch.model.vpe.ViewpointEncoding",
    "create_backbone": "CanViTConfig.backbone_name (CanViT builds its own backbone)",
    "CLSStandardizer": "canvit_pytorch.model.standardizer.PositionAwareStandardizer",
    "PatchStandardizer": "canvit_pytorch.model.standardizer.PositionAwareStandardizer",
    "PositionAwareStandardizer": "canvit_pytorch.model.standardizer.PositionAwareStandardizer",
    "CANVIT_REPO_ROOT": "canvit_pytorch.hub.repos.HUB_ROOT, set by the environment variable CANVIT_HUB_ROOT",
    "resolve_canvit_repo": "canvit_pytorch.hub.repos.hub_repo",
    "fuse_probe": "canvit_pytorch.model.classification.fuse_probe",
}
REMOVED = {
    "CanViTForRGBReconstruction", "CanViTForRGBReconstructionHFHub", "make_rgb_repo_id",
    "patchify", "unpatchify", "load_canvit_base",
}


def attribute_error(module: str, name: str) -> AttributeError:
    if name in RENAMED:
        return AttributeError(f"{module}.{name} is now {RENAMED[name]}. {UPGRADE_HINT}")
    if name in REMOVED:
        return AttributeError(f"{module}.{name} was removed in canvit-pytorch 0.2 (RGB-reconstruction pretraining). {UPGRADE_HINT}")
    return AttributeError(f"module {module!r} has no attribute {name!r}")


def _read_config(model_id: str, revision: str | None) -> dict:
    path = Path(model_id) / "config.json" if Path(model_id).is_dir() else hf_hub_download(model_id, "config.json", revision=revision)
    return json.loads(Path(path).read_text())


def checkpoint_format_error(model_id: str, *, revision: str | None, cls: type, missing: list[str]) -> ValueError:
    config = _read_config(model_id, revision)
    if "model_config" in config and "backbone_name" in config:
        return ValueError(f"{model_id} is a canvit-pytorch 0.1 checkpoint. {UPGRADE_HINT}")
    return ValueError(f"{model_id}: config.json lacks {missing}, required by {cls.__name__}; it has {sorted(config)}")


def training_checkpoint_error(path: Path) -> ValueError:
    return ValueError(
        f"{path} is a training checkpoint written by canvit-pytorch 0.1, whose optimizer state is ordered by the "
        "0.1 parameter layout; resume that run with canvit-pytorch<0.2."
    )


def weights_mismatch_error(model_file: str, *, missing: list[str], unexpected: list[str]) -> ValueError:
    return ValueError(
        f"Weights in {model_file} do not match the model: {len(missing)} missing, e.g. {sorted(missing)[:3]}; "
        f"{len(unexpected)} unexpected, e.g. {sorted(unexpected)[:3]}. {UPGRADE_HINT}"
    )


# 0.1 → 0.2 weight names, applied in order to each key.
_CANVIT_KEY_RULES: list[tuple[str, str]] = [
    (r"^canvas_read\.", "canvas_reads."),
    (r"^canvas_write\.", "canvas_writes."),
    (r"^(canvas_(?:reads|writes)\.\d+)\.q_proj\.", r"\1.q_map."),
    (r"^(canvas_(?:reads|writes)\.\d+)\.k_proj\.", r"\1.k_map."),
    (r"^(canvas_(?:reads|writes)\.\d+)\.v_proj\.", r"\1.v_map."),
    (r"^(canvas_(?:reads|writes)\.\d+)\.out_proj\.", r"\1.o_map."),
    (r"^(canvas_(?:reads|writes)\.\d+)\.q_norm\.", r"\1.ln_q."),
    (r"^(canvas_(?:reads|writes)\.\d+)\.kv_norm\.", r"\1.ln_kv."),
    (r"^canvas_register_init$", "init_canvas_registers"),
    (r"^canvas_spatial_init$", "init_canvas_patch"),
    (r"^recurrent_cls_init$", "init_recurrent_cls"),
    (r"^vpe\.B$", "vpe.frequencies"),
]
_PRETRAINING_KEY_RULES: list[tuple[str, str]] = [
    (r"^scene_patches_head\.", "teacher_patch_readout."),
    (r"^scene_cls_head\.", "teacher_cls_readout."),
    (r"^scene_standardizers\.\d+\._initialized$", "teacher_patch_standardizer.fitted"),
    (r"^scene_standardizers\.\d+\.", "teacher_patch_standardizer."),
    (r"^cls_standardizers\.\d+\._initialized$", "teacher_cls_standardizer.fitted"),
    (r"^cls_standardizers\.\d+\.", "teacher_cls_standardizer."),
]
_CLASSIFICATION_KEY_RULES: list[tuple[str, str]] = [(r"^norm\.", "readout.norm."), (r"^head\.", "readout.proj.")]


def _rename(key: str, rules: list[tuple[str, str]]) -> str:
    for pattern, replacement in rules:
        key = re.sub(pattern, replacement, key)
    return key


def convert_canvit_config(backbone_name: str, model_config: dict) -> dict:
    fields = dict(model_config)
    assert fields.pop("canvas_update_mode", "additive") == "additive", model_config
    assert fields.pop("gate_bias_init", None) is None, model_config
    fields.pop("teacher_dim", None)
    converted = {
        "backbone_name": backbone_name,
        "canvas_num_heads": fields.pop("canvas_num_heads"),
        "canvas_head_dim": fields.pop("canvas_head_dim"),
        "num_canvas_registers": fields.pop("n_canvas_registers"),
        "num_backbone_registers": fields.pop("n_backbone_registers"),
        "rw_stride": fields.pop("rw_stride"),
        "enable_reads": fields.pop("enable_reads", True),
        "enable_vpe": fields.pop("enable_vpe"),
        "canvas_projections": {"asymmetric": "asymmetric", "full": "qkvo"}[fields.pop("canvas_proj_mode", "asymmetric")],
    }
    assert not fields, f"unconverted 0.1 config fields: {fields}"
    return converted


def convert_state_dict(state_dict: dict[str, Tensor], *, kind: str) -> dict[str, Tensor]:
    """kind: "pretraining" (a 0.1 CanViTForPretraining) or "classification" (a 0.1 CanViTForImageClassification)."""
    converted = {}
    for key, value in state_dict.items():
        if kind == "pretraining":
            new = _rename(key, _PRETRAINING_KEY_RULES)
            new = new if new != key else "canvit." + _rename(key, _CANVIT_KEY_RULES)
        elif kind == "classification":
            new = _rename(key, _CLASSIFICATION_KEY_RULES)
            new = new if new != key else "canvit." + _rename(key.removeprefix("canvit."), _CANVIT_KEY_RULES)
        else:
            raise ValueError(kind)
        converted[new] = value
    assert len(converted) == len(state_dict), "two 0.1 weights mapped to one 0.2 name"
    return converted


def convert_checkpoint(source: str, out_dir: Path) -> Path:
    """Rewrite a 0.1 CanViTForPretraining or CanViTForImageClassification checkpoint (Hub repo or directory)."""
    config = _read_config(source, None)
    weights_path = Path(source) / "model.safetensors" if Path(source).is_dir() else hf_hub_download(source, "model.safetensors")
    state_dict = load_file(weights_path)
    extras = {k: v for k, v in config.items() if k in ("metadata", "training")}
    canvit_config = convert_canvit_config(config["backbone_name"], config["model_config"])
    if "canvas_patch_grid_sizes" in config:
        (grid,) = config["canvas_patch_grid_sizes"]
        new_config = {"canvit_config": canvit_config, "teacher_dim": config["model_config"]["teacher_dim"],
                      "teacher_patch_grid": grid, **extras}
        state_dict = convert_state_dict(state_dict, kind="pretraining")
    else:
        new_config = {"canvit_config": canvit_config, "n_classes": config["n_classes"], **extras}
        state_dict = convert_state_dict(state_dict, kind="classification")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(new_config, indent=2, sort_keys=True) + "\n")
    save_file(state_dict, out_dir / "model.safetensors")
    return out_dir


def main(source: str, out_dir: Path, /) -> None:
    """Convert a canvit-pytorch 0.1 checkpoint (Hub repo id or local directory) to the 0.2 format."""
    print(convert_checkpoint(source, out_dir))


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
