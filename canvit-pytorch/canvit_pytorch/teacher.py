"""The frozen DINOv3 models CanViT distills from and is compared against, loaded from the Hugging Face Hub."""

from typing import Literal, NamedTuple

import torch
from torch import Tensor, nn
from transformers import AutoModel, PreTrainedModel

DINOv3Variant = Literal["vits16", "vitb16"]

DINOV3_REPOS: dict[DINOv3Variant, str] = {
    "vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
}
DINOV3_NAMES: dict[DINOv3Variant, str] = {"vits16": "DINOv3 ViT-S/16", "vitb16": "DINOv3 ViT-B/16"}
DINOV3_PATCH_SIZE = 16
# CanViT-B's pretraining teacher.
TEACHER_REPO = DINOV3_REPOS["vitb16"]


class TeacherFeatures(NamedTuple):
    patches: Tensor  # [B, H*W, D], after DINOv3's final LayerNorm
    cls: Tensor  # [B, D], after DINOv3's final LayerNorm


class DINOv3Teacher(nn.Module):
    NUM_PREFIX_TOKENS = 5  # CLS and 4 registers

    def __init__(self, model: PreTrainedModel) -> None:
        super().__init__()
        self.model = model

    @property
    def embed_dim(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def patch_size(self) -> int:
        return int(self.model.config.patch_size)

    def forward(self, images: Tensor) -> TeacherFeatures:
        """ImageNet-normalized [B, 3, H, W] -> patch and CLS features."""
        tokens = self.model(images).last_hidden_state
        assert tokens.shape[1] > self.NUM_PREFIX_TOKENS, tokens.shape
        return TeacherFeatures(patches=tokens[:, self.NUM_PREFIX_TOKENS :], cls=tokens[:, 0])


def load_teacher(repo_id: str, device: torch.device) -> DINOv3Teacher:
    """A frozen DINOv3 in evaluation mode, float32 weights."""
    model = AutoModel.from_pretrained(repo_id, dtype=torch.float32)
    assert isinstance(model, PreTrainedModel)
    model.eval().requires_grad_(False)
    return DINOv3Teacher(model).to(device)
