"""Checkpoint serialization for CanViTForPretraining models."""

import logging
import os
import re
import socket
import subprocess
import sys
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

import dacite
import torch
from canvit_pytorch import CanViT
from torch import Tensor

from canvit_pretrain import CanViTForPretraining, CanViTForPretrainingConfig

log = logging.getLogger(__name__)

class CheckpointData(TypedDict):
    """Checkpoint structure. All fields present; None where not applicable.

    Teacher fields are None for the teacher-free ``rgb_recon`` objective.
    ``objective`` is absent in pre-objective checkpoints and defaults to "distillation".
    """

    # --- Model reconstruction (required) ---
    state_dict: dict[str, Tensor]
    model_config: dict
    backbone_name: str
    objective: str
    canvas_patch_grid_sizes: list[int]
    teacher_dim: int | None
    teacher_repo_id: str | None
    teacher_name: str | None
    dataset: str

    # --- Training context ---
    glimpse_grid_size: int
    scene_resolution: int
    step: int | None
    train_loss: float | None

    # --- Optimizer/scheduler state for resuming training ---
    optimizer_state: dict | None
    scheduler_state: dict | None
    training_config_history: dict[str, dict] | None

    # --- Provenance (last save only — see provenance_history for full trail) ---
    timestamp: str
    git_commit: str | None
    git_dirty: bool
    comet_id: str | None
    hostname: str | None
    slurm_job_id: str | None
    slurm_array_task_id: str | None
    cmdline: list[str] | None

    # --- Provenance history (accumulated across resumes) ---
    provenance_history: dict[str, dict] | None


def _git_info() -> tuple[str | None, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL) != 0
        return commit, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, False


def get_env_metadata() -> tuple[str | None, str | None, str | None, list[str] | None]:
    """Collect (hostname, slurm_job_id, slurm_array_task_id, cmdline). Best-effort."""
    hostname: str | None = None
    cmdline: list[str] | None = None
    try:
        hostname = socket.gethostname()
    except Exception as e:
        log.warning(f"Failed to get hostname: {e}")
    try:
        cmdline = sys.argv.copy()
    except Exception as e:
        log.warning(f"Failed to get cmdline: {e}")
    return (
        hostname,
        os.environ.get("SLURM_JOB_ID"),
        os.environ.get("SLURM_ARRAY_TASK_ID"),
        cmdline,
    )


def current_provenance() -> dict:
    """Snapshot of current environment provenance (git, host, slurm, cmdline)."""
    git_commit, git_dirty = _git_info()
    hostname, slurm_job_id, slurm_array_task_id, cmdline = get_env_metadata()
    return {
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "hostname": hostname,
        "slurm_job_id": slurm_job_id,
        "slurm_array_task_id": slurm_array_task_id,
        "cmdline": cmdline,
    }


def atomic_torch_save(data: CheckpointData, path: Path) -> None:
    """Save data to path atomically using tmp file + rename."""
    log.info(f"Saving checkpoint to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path_str = tempfile.mkstemp(
        suffix=".pt.tmp", prefix=path.stem, dir=path.parent
    )
    tmp_path = Path(tmp_path_str)
    log.debug(f"Writing to tmp file: {tmp_path}")
    try:
        os.close(fd)
        torch.save(data, tmp_path)
        tmp_path.rename(path)
    except Exception:
        log.exception(f"Failed to save checkpoint to {path}")
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def update_symlink(symlink_path: Path, target: Path) -> None:
    """Atomically update symlink to point to target."""
    log.info(f"Updating symlink {symlink_path} -> {target.name}")
    tmp_link = symlink_path.parent / f".{symlink_path.name}.tmp.{os.getpid()}"
    if tmp_link.is_symlink() or tmp_link.exists():
        tmp_link.unlink()
    tmp_link.symlink_to(target.name)
    tmp_link.rename(symlink_path)


def find_latest(run_dir: Path) -> Path | None:
    """Find latest.pt symlink in run_dir, return resolved path or None."""
    latest = run_dir / "latest.pt"
    if latest.is_symlink():
        resolved = latest.resolve()
        if resolved.exists():
            return resolved
        log.warning(f"latest.pt symlink broken: {latest} -> {resolved}")
    return None



def save(
    path: Path,
    model: CanViT,
    backbone_name: str,
    *,
    objective: str = "distillation",
    teacher_repo_id: str | None = None,
    teacher_name: str | None = None,
    dataset: str,
    glimpse_grid_size: int,
    scene_resolution: int,
    canvas_patch_grid_sizes: list[int] | None = None,
    step: int | None = None,
    train_loss: float | None = None,
    comet_id: str | None = None,
    optimizer_state: dict | None = None,
    scheduler_state: dict | None = None,
    training_config_history: dict[str, dict] | None = None,
    provenance_history: dict[str, dict] | None = None,
) -> None:
    """Save checkpoint with all info needed to reconstruct model and push to hub.

    Teacher fields are required for ``distillation`` and unused for ``rgb_recon``
    (which has no standardizers; ``canvas_patch_grid_sizes`` must be passed explicitly).
    """
    if objective == "distillation":
        assert isinstance(model, CanViTForPretraining)
        assert isinstance(model.cfg, CanViTForPretrainingConfig)
        assert teacher_repo_id is not None and teacher_name is not None, "distillation save needs teacher fields"
        teacher_dim: int | None = model.cfg.teacher_dim
        grids = canvas_patch_grid_sizes if canvas_patch_grid_sizes is not None else model.canvas_patch_grid_sizes
    else:
        teacher_dim = None
        assert canvas_patch_grid_sizes is not None, "rgb_recon save needs canvas_patch_grid_sizes"
        grids = canvas_patch_grid_sizes
    git_commit, git_dirty = _git_info()
    hostname, slurm_job_id, slurm_array_task_id, cmdline = get_env_metadata()

    data: CheckpointData = {
        "state_dict": model.state_dict(),
        "model_config": asdict(model.cfg),
        "backbone_name": backbone_name,
        "objective": objective,
        "canvas_patch_grid_sizes": grids,
        "teacher_dim": teacher_dim,
        "teacher_repo_id": teacher_repo_id,
        "teacher_name": teacher_name,
        "dataset": dataset,
        "glimpse_grid_size": glimpse_grid_size,
        "scene_resolution": scene_resolution,
        "step": step,
        "train_loss": train_loss,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "training_config_history": training_config_history,
        "provenance_history": provenance_history,
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "comet_id": comet_id,
        "hostname": hostname,
        "slurm_job_id": slurm_job_id,
        "slurm_array_task_id": slurm_array_task_id,
        "cmdline": cmdline,
    }

    atomic_torch_save(data, path)
    size_mb = path.stat().st_size / (1024 * 1024)

    log.info(f"Checkpoint saved: {path} ({size_mb:.1f} MB)")
    log.info(
        f"  backbone={backbone_name}, objective={objective}, canvas_patch_grid_sizes={grids},"
        f" teacher={teacher_name}, dataset={dataset},"
        f" glimpse={glimpse_grid_size}, scene={scene_resolution}px"
    )
    if step is not None:
        log.info(f"  step={step}, train_loss={train_loss:.4e}" if train_loss else f"  step={step}")
    if git_commit:
        log.info(f"  git={git_commit[:8]}{'*' if git_dirty else ''}")


def load(path: Path, device: torch.device | str = "cpu") -> CheckpointData:
    """Load checkpoint data. Teacher fields are required only for distillation checkpoints."""
    log.info(f"Loading checkpoint: {path}")
    raw = torch.load(path, weights_only=False, map_location=device)

    # Pre-objective checkpoints (flagship, ablations) have no "objective" key.
    objective = raw.get("objective", "distillation")

    required = ["state_dict", "model_config", "backbone_name", "dataset"]
    if objective == "distillation":
        required += ["canvas_patch_grid_sizes", "teacher_dim", "teacher_repo_id", "teacher_name"]
    for key in required:
        assert key in raw, f"Checkpoint {path.name} missing required field: {key!r}"

    data: CheckpointData = {
        "state_dict": raw["state_dict"],
        "model_config": raw["model_config"],
        "backbone_name": raw["backbone_name"],
        "objective": objective,
        "canvas_patch_grid_sizes": raw.get("canvas_patch_grid_sizes", []),
        "teacher_dim": raw.get("teacher_dim"),
        "teacher_repo_id": raw.get("teacher_repo_id"),
        "teacher_name": raw.get("teacher_name"),
        "dataset": raw["dataset"],
        "glimpse_grid_size": raw["glimpse_grid_size"],
        "scene_resolution": raw["scene_resolution"],
        "step": raw["step"],
        "train_loss": raw["train_loss"],
        "optimizer_state": raw["optimizer_state"],
        "scheduler_state": raw["scheduler_state"],
        "training_config_history": raw["training_config_history"],
        "provenance_history": raw.get("provenance_history"),
        "timestamp": raw["timestamp"],
        "git_commit": raw["git_commit"],
        "git_dirty": raw["git_dirty"],
        "comet_id": raw["comet_id"],
        "hostname": raw["hostname"],
        "slurm_job_id": raw["slurm_job_id"],
        "slurm_array_task_id": raw["slurm_array_task_id"],
        "cmdline": raw["cmdline"],
    }

    log.info(
        f"  backbone={data['backbone_name']}, grids={data['canvas_patch_grid_sizes']},"
        f" teacher={data['teacher_name']}, dataset={data['dataset']},"
        f" scene={data['scene_resolution']}px"
    )
    if data["step"] is not None:
        step = data["step"]
        msg = f"  step={step}, train_loss={data['train_loss']:.4e}" if data["train_loss"] else f"  step={step}"
        log.info(msg)
    if data["git_commit"]:
        log.info(f"  git={data['git_commit'][:8]}{'*' if data['git_dirty'] else ''}")

    return data


_STANDARDIZER_RE = re.compile(r"(cls|scene)_standardizers\.")


def load_state_dict_flexible(
    model: CanViTForPretraining,
    state_dict: dict[str, Tensor],
) -> None:
    """Load state dict, allowing standardizer key mismatches for grid-size changes.

    All non-standardizer keys must match exactly. Standardizer keys may differ
    when the canvas grid size changed between checkpoint and current model.
    """
    result = model.load_state_dict(state_dict, strict=False)
    bad_missing = [k for k in result.missing_keys if not _STANDARDIZER_RE.match(k)]
    bad_unexpected = [k for k in result.unexpected_keys if not _STANDARDIZER_RE.match(k)]
    assert not bad_missing, f"Missing core weights: {bad_missing}"
    assert not bad_unexpected, f"Unexpected core weights: {bad_unexpected}"
    if result.missing_keys or result.unexpected_keys:
        log.warning(f"Standardizer key mismatch (grid size change): "
                    f"missing={result.missing_keys}, unexpected={result.unexpected_keys}")
    else:
        log.info("Model state loaded (all keys matched)")


def load_model(
    path: Path, device: torch.device | str = "cpu",
) -> tuple[CanViT, CheckpointData]:
    """Load the pretraining model (distillation or RGB) from checkpoint. Returns (model, data)."""
    from typing import cast

    from canvit_pytorch import CanViTConfig, CanViTForRGBReconstruction, create_backbone
    from canvit_pytorch.backbone import BackboneName

    ckpt = load(path, device)
    backbone_name = cast(BackboneName, ckpt["backbone_name"])

    model: CanViT
    if ckpt["objective"] == "distillation":
        cfg = dacite.from_dict(CanViTForPretrainingConfig, ckpt["model_config"])
        model = CanViTForPretraining(
            backbone=create_backbone(backbone_name),
            cfg=cfg,
            backbone_name=backbone_name,
            canvas_patch_grid_sizes=ckpt["canvas_patch_grid_sizes"],
        )
        load_state_dict_flexible(model, ckpt["state_dict"])
    else:
        rgb_cfg = dacite.from_dict(CanViTConfig, ckpt["model_config"])
        model = CanViTForRGBReconstruction(
            backbone=create_backbone(backbone_name),
            cfg=rgb_cfg,
            backbone_name=backbone_name,
        )
        model.load_state_dict(ckpt["state_dict"])

    if isinstance(device, str):
        device = torch.device(device)
    return model.to(device).eval(), ckpt
