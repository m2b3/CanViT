"""Per-objective loss + metric builders fed to :func:`training_step`.

Two pretraining objectives:

- **Distillation** (flagship): reconstruct DINOv3 teacher patch + CLS features. Replicates
  the original ``compute_loss`` / branch-metric behavior exactly.
- **RGB reconstruction** (teacher-free control): reconstruct raw pixel patches from the
  canvas. Like MAE, patches only — no CLS term.

Each builder closes over the targets + model and returns callables matching the
``LossFn`` / ``BranchMetricsFn`` contracts in :mod:`canvit_pretrain.train.step`.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from canvit_pytorch import (
    CanViTForPretraining,
    CanViTForRGBReconstruction,
    CanViTOutput,
    RecurrentState,
)
from torch import Tensor

from .step import BranchMetricsFn, LossFn, LossOutput


def distillation_loss_fn(
    *,
    model: CanViTForPretraining,
    scene_target: Tensor,
    cls_target: Tensor,
    enable_scene_patches_loss: bool,
    enable_scene_cls_loss: bool,
) -> LossFn:
    """Teacher-feature MSE on canvas patches (+ CLS). Targets are pre-standardized."""

    def loss_fn(out: CanViTOutput) -> LossOutput:
        components: dict[str, Tensor] = {}
        if enable_scene_patches_loss:
            scene_pred = model.predict_teacher_scene(out.state.canvas)
            components["scene_patches_loss"] = F.mse_loss(scene_pred, scene_target)
        if enable_scene_cls_loss:
            cls_pred = model.predict_scene_teacher_cls(out.state.recurrent_cls)
            components["scene_cls_loss"] = F.mse_loss(cls_pred, cls_target)
        assert components, "At least one distillation loss must be enabled"
        combined = torch.stack(list(components.values())).sum()
        return LossOutput(combined=combined, components=components)

    return loss_fn


def distillation_branch_metrics_fn(
    *,
    model: CanViTForPretraining,
    scene_target: Tensor,
    cls_target: Tensor,
    raw_scene_target: Tensor,
    raw_cls_target: Tensor,
    scene_denorm: Callable[[Tensor], Tensor],
    cls_denorm: Callable[[Tensor], Tensor],
) -> BranchMetricsFn:
    """Cosine similarity of the final-state predictions to teacher features (raw + normalized)."""

    def fn(state: RecurrentState) -> dict[str, Tensor]:
        scene_pred = model.predict_teacher_scene(state.canvas)
        cls_pred = model.predict_scene_teacher_cls(state.recurrent_cls)
        scene_pred_raw = scene_denorm(scene_pred)
        cls_pred_raw = cls_denorm(cls_pred.unsqueeze(1)).squeeze(1)
        return {
            "scene_cos_raw": F.cosine_similarity(scene_pred_raw, raw_scene_target, dim=-1).mean(),
            "scene_cos_norm": F.cosine_similarity(scene_pred, scene_target, dim=-1).mean(),
            "cls_cos_raw": F.cosine_similarity(cls_pred_raw, raw_cls_target, dim=-1).mean(),
            "cls_cos_norm": F.cosine_similarity(cls_pred, cls_target, dim=-1).mean(),
        }

    return fn


def rgb_loss_fn(*, model: CanViTForRGBReconstruction, pixel_target: Tensor) -> LossFn:
    """Pixel-MSE on per-patch RGB reconstructed from the canvas. ``pixel_target`` is patchified."""

    def loss_fn(out: CanViTOutput) -> LossOutput:
        pred = model.predict_rgb_patches(out.state.canvas)
        recon = F.mse_loss(pred, pixel_target)
        return LossOutput(combined=recon, components={"recon_loss": recon})

    return loss_fn


def rgb_branch_metrics_fn(*, model: CanViTForRGBReconstruction, pixel_target: Tensor) -> BranchMetricsFn:
    """No extra end-of-branch metrics for RGB; the averaged recon_loss component is the signal."""

    def fn(state: RecurrentState) -> dict[str, Tensor]:
        return {}

    return fn
