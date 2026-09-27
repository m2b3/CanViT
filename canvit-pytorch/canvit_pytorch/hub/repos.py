"""Names of released CanViT checkpoints and probes on the Hugging Face Hub.

Names resolve under HUB_ROOT: the `canvit` organization, or a local directory
laid out the same way when $CANVIT_HUB_ROOT points to one.
"""

import os
from typing import Literal

from canvit_pytorch.project import HUB_ORGANIZATION

HUB_ROOT = os.environ.get("CANVIT_HUB_ROOT", HUB_ORGANIZATION).rstrip("/")


def hub_repo(name: str) -> str:
    return f"{HUB_ROOT}/{name}"


PretrainingDataset = Literal["in21k", "in1k"]

# CanViT-B pretrained by distillation from DINOv3 ViT-B on each dataset. IN21k is
# the paper's model; IN1k-only pretraining answered a question on dataset scale.
PRETRAINED: dict[PretrainingDataset, str] = {
    "in21k": hub_repo("canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02"),
    "in1k": hub_repo("canvitb16-add-vpe-pretrain-g128px-s512px-in1k-dv3b16-2026-06-22"),
}
FLAGSHIP = PRETRAINED["in21k"]

# The viewing geometry every released CanViT-B was pretrained with ("g128px-s512px" in the names): 128 px
# glimpses of 512 px scenes and a 32×32 canvas, one canvas patch per 16 px. Pretraining defaults to it,
# and the paper's evaluations, fine-tuning and visualizations reuse it.
RELEASED_GLIMPSE_SIZE_PX = 128
RELEASED_SCENE_SIZE_PX = 512
RELEASED_CANVAS_GRID_SIZE = 32

# The Hub tag on each checkpoint released before canvit-pytorch 0.2: its files in the format 0.1 code reads.
OLD_FORMAT_REVISION = "canvit-pytorch-0.1"

# The flagship fine-tuned on ImageNet-1k classification (LP-FT, TPU).
FINETUNED_IN1K = hub_repo("canvitb16-add-vpe-finetune-g128px-s512px-in1k-2026-04-06")

# The same LP-FT recipe run from each pretrained checkpoint with a JAX/Flax NNX trainer, exported to PyTorch.
FINETUNED_IN1K_NNX: dict[PretrainingDataset, str] = {
    "in21k": hub_repo("canvitb16-add-vpe-finetune-g128px-s512px-in1k-2026-07-24"),
    "in1k": hub_repo("canvitb16-add-vpe-finetune-g128px-s512px-in1k-from-in1k-2026-07-24"),
}

# Linear ImageNet-1k probe on DINOv3 ViT-B/16's CLS token at 512 px (from m2b3/dinov3-in1k-probes);
# fused into CanViT's CLS readout for frozen classification.
DINOV3_VITB16_IN1K_PROBE = hub_repo("dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe")

# How probe names refer to the DINOv3 teachers (keys: canvit_pytorch.teacher.DINOV3_REPOS).
DINOV3_PROBE_MODEL_NAMES = {"vitb16": "dv3b", "vits16": "dv3s"}

# Every released ADE20K probe was trained for this many steps; the count is part of its name.
RELEASED_PROBE_STEPS = 40_000


def ade20k_probe_name(model: str, *, scene_size_px: int, canvas_grid_size: int, steps: int) -> str:
    """Name of a probe on the canvas of `model` (a PRETRAINED key or `abl-<ablation>`)."""
    assert steps % 1000 == 0, steps
    return f"probe-ade20k-{steps // 1000}k-s{scene_size_px}-c{canvas_grid_size}-{model}"


def dinov3_ade20k_probe_name(model: str, *, input_size_px: int, steps: int) -> str:
    """Name of a probe on DINOv3 patch features; `model` is dv3b (ViT-B/16) or dv3s (ViT-S/16)."""
    assert steps % 1000 == 0, steps
    return f"probe-ade20k-{steps // 1000}k-{model}-{input_size_px}px"


def released_ade20k_probe(model: str, *, scene_size_px: int, canvas_grid_size: int) -> str:
    name = ade20k_probe_name(model, scene_size_px=scene_size_px, canvas_grid_size=canvas_grid_size, steps=RELEASED_PROBE_STEPS)
    return hub_repo(name)


def released_dinov3_ade20k_probe(model: str, *, input_size_px: int) -> str:
    return hub_repo(dinov3_ade20k_probe_name(model, input_size_px=input_size_px, steps=RELEASED_PROBE_STEPS))
