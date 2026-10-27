"""Released CanViT checkpoints in the canvit-nnx format on the Hugging Face Hub, and their viewing geometry.

canvit_pytorch.hub.repos names the same releases for PyTorch; each NNX
repository is the PyTorch one with an "-nnx" suffix.
"""

from typing import Literal

PretrainingDataset = Literal["in21k", "in1k"]

PRETRAINED: dict[PretrainingDataset, str] = {
    "in21k": "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02-nnx",
    "in1k": "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in1k-dv3b16-2026-06-22-nnx",
}
"""CanViT-B pretrained by distillation from DINOv3 ViT-B on each dataset."""

DINOV3_VITB16_IN1K_PROBE = "canvit/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe"
"""Linear ImageNet-1k probe on DINOv3 ViT-B/16's CLS token at 512 px; framework-independent safetensors."""

RELEASED_GLIMPSE_SIZE_PX = 128
RELEASED_SCENE_SIZE_PX = 512
RELEASED_CANVAS_GRID_SIZE = 32

__all__ = [
    "DINOV3_VITB16_IN1K_PROBE",
    "PRETRAINED",
    "RELEASED_CANVAS_GRID_SIZE",
    "RELEASED_GLIMPSE_SIZE_PX",
    "RELEASED_SCENE_SIZE_PX",
    "PretrainingDataset",
]
