"""The distillation objective (paper, Section 5.1) and how close the predictions come to the teacher.

After every glimpse, CanViT predicts the teacher's features of the whole scene:
its patch features from the canvas and its CLS token from the recurrent CLS
token, both standardized per position.
"""

from dataclasses import dataclass

import torch.nn.functional as F
from torch import Tensor

from canvit_pytorch.model.canvit import RecurrentState
from canvit_pytorch.model.pretraining import CanViTForPretraining


@dataclass(frozen=True)
class DistillationTargets:
    patches: Tensor  # [B, G*G, D], standardized per position
    cls: Tensor  # [B, D], standardized
    raw_patches: Tensor  # [B, G*G, D], as the teacher outputs them
    raw_cls: Tensor  # [B, D]

    @staticmethod
    def standardize(model: CanViTForPretraining, *, patches: Tensor, cls: Tensor) -> "DistillationTargets":
        """Targets from the teacher's float32 features, standardized with the model's fitted statistics."""
        return DistillationTargets(
            patches=model.teacher_patch_standardizer(patches),
            cls=model.teacher_cls_standardizer(cls.unsqueeze(1)).squeeze(1),
            raw_patches=patches,
            raw_cls=cls,
        )


def distillation_losses(
    model: CanViTForPretraining, state: RecurrentState, targets: DistillationTargets,
    *, enable_teacher_patch_loss: bool, enable_teacher_cls_loss: bool,
) -> dict[str, Tensor]:
    """The enabled mean squared errors, by name."""
    losses: dict[str, Tensor] = {}
    if enable_teacher_patch_loss:
        losses["teacher_patch_loss"] = F.mse_loss(model.predict_teacher_patches(state.canvas), targets.patches)
    if enable_teacher_cls_loss:
        losses["teacher_cls_loss"] = F.mse_loss(model.predict_teacher_cls(state.recurrent_cls), targets.cls)
    assert losses, "enable at least one loss"
    return losses


def teacher_similarities(model: CanViTForPretraining, state: RecurrentState, targets: DistillationTargets) -> dict[str, Tensor]:
    """Mean cosine similarity of the predictions to the teacher's features, in standardized and raw space."""
    patches = model.predict_teacher_patches(state.canvas)
    cls = model.predict_teacher_cls(state.recurrent_cls)
    raw_patches = model.teacher_patch_standardizer.destandardize(patches)
    raw_cls = model.teacher_cls_standardizer.destandardize(cls.unsqueeze(1)).squeeze(1)
    return {
        "patch_cos_standardized": F.cosine_similarity(patches, targets.patches, dim=-1).mean(),
        "patch_cos_raw": F.cosine_similarity(raw_patches, targets.raw_patches, dim=-1).mean(),
        "cls_cos_standardized": F.cosine_similarity(cls, targets.cls, dim=-1).mean(),
        "cls_cos_raw": F.cosine_similarity(raw_cls, targets.raw_cls, dim=-1).mean(),
    }
