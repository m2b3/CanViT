"""Why code or checkpoints written for canvit-pytorch 0.1 fail, and what to do.

canvit-pytorch 0.2 renamed public classes, checkpoint config keys and weight
names. The package calls into this module only after a lookup or a load has
failed.
"""

import json
from pathlib import Path

from canvit_core.checkpoint import hub_file

from canvit_pytorch.hub.repos import OLD_FORMAT_REVISION

UPGRADE_HINT = (
    "canvit-pytorch 0.2 changed its API and checkpoint format, and the released checkpoints on the Hub are in "
    f"the new format. Code written for 0.1 needs `canvit-pytorch<0.2` and the checkpoints' revision "
    f"`{OLD_FORMAT_REVISION}`."
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
    "fuse_probe": "canvit_core.readout.fuse_probe",
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


def checkpoint_format_error(model_id: str, *, revision: str | None, cls: type, missing: list[str]) -> ValueError:
    config = json.loads(hub_file(model_id, "config.json", revision=revision).read_text())
    if "model_config" in config and "backbone_name" in config:
        return ValueError(f"{model_id} is a canvit-pytorch 0.1 checkpoint. {UPGRADE_HINT}")
    return ValueError(f"{model_id}: config.json lacks {missing}, required by {cls.__name__}; it has {sorted(config)}")


def training_checkpoint_error(path: Path) -> ValueError:
    return ValueError(
        f"{path} is a training checkpoint written by canvit-pytorch 0.1, whose optimizer state is ordered by the "
        "0.1 parameter layout; resume that run with canvit-pytorch<0.2."
    )
