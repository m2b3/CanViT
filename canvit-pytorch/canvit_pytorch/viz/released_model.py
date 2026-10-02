"""The released CanViT-B with its ADE20K probe, as the web visualizations load it, and the manifest records naming both."""

from dataclasses import dataclass
from typing import Any

import torch
from huggingface_hub import HfApi

from canvit_pytorch.benchmarks.ade20k import CLASS_NAMES
from canvit_pytorch.hub.repos import (
    FLAGSHIP,
    RELEASED_GLIMPSE_SIZE_PX,
    RELEASED_SCENE_SIZE_PX,
    VIZ_CANVAS_GRID_SIZE,
    released_ade20k_probe,
)
from canvit_pytorch.model.segmentation import CanViTForSemanticSegmentation
from canvit_pytorch.project import HUB_ORGANIZATION

# The released model's pretraining geometry (g128px-s512px in its name), and the canvas grid of the
# probe the project page shows.
SCENE_SIZE_PX = RELEASED_SCENE_SIZE_PX
GLIMPSE_SIZE_PX = RELEASED_GLIMPSE_SIZE_PX
CANVAS_GRID_SIZE = VIZ_CANVAS_GRID_SIZE


def hub_identity(repo: str) -> dict[str, str]:
    """The Hub name and current revision of a repo, also when it was loaded from a local copy (CANVIT_HUB_ROOT)."""
    name = f"{HUB_ORGANIZATION}/{repo.rsplit('/', 1)[-1]}"
    revision = HfApi().model_info(name).sha
    assert revision, f"no Hub revision for {name}"
    return {"repo": name, "revision": revision}


@dataclass(frozen=True)
class ReleasedSegmenter:
    model: CanViTForSemanticSegmentation
    model_record: dict[str, Any]
    """A manifest's "model": the checkpoint's Hub identity and the architecture facts renderers show."""
    readout_record: dict[str, Any]
    """A manifest's "readout": the probe's Hub identity and its classes, in output order."""


def load_released_segmenter(*, scene_size_px: int, canvas_grid_size: int, device: torch.device) -> ReleasedSegmenter:
    probe_repo = released_ade20k_probe("in21k", scene_size_px=scene_size_px, canvas_grid_size=canvas_grid_size)
    model = CanViTForSemanticSegmentation.from_pretrained_with_probe(pretrained_repo=FLAGSHIP, probe_repo=probe_repo)
    model = model.to(device).eval()
    canvit = model.canvit
    model_record = {
        **hub_identity(FLAGSHIP),
        "backbone": canvit.config.backbone_name, "num_blocks": len(canvit.backbone.blocks),
        "patch_px": canvit.patch_size, "canvas_dim": canvit.canvas_dim,
        "read_after_blocks": list(canvit.read_after_blocks), "write_after_blocks": list(canvit.write_after_blocks),
    }
    readout_record = {"kind": "ade20k-segmentation", **hub_identity(probe_repo),
                      "num_classes": len(CLASS_NAMES), "class_names": list(CLASS_NAMES)}
    return ReleasedSegmenter(model=model, model_record=model_record, readout_record=readout_record)
