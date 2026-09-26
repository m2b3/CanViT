"""Validation with streaming metrics and optional PCA visualization."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import comet_ml
import numpy as np
import torch
import torch.nn.functional as F
from canvit_pytorch import CanViTOutput, CLSStandardizer, PatchStandardizer, RecurrentState
from canvit_pytorch.backbone.vit import NormFeatures
from canvit_pytorch.teacher import DINOv3Teacher
from canvit_pytorch.viewpoint import Viewpoint as CanvitViewpoint
from dinov3_in1k_probes import DINOv3LinearClassificationHead
from torch import Tensor

from canvit_pretrain import CanViTForPretraining
from canvit_pretrain.train.utils import assert_shape

from ..probe import (
    compute_in1k_top1,
    get_imagenet_class_names,
    get_probe_resolution,
    get_top_k_predictions,
    labels_are_in1k,
)
from ..viewpoint import make_eval_viewpoints
from .comet import log_curve, log_figure
from .image import imagenet_denormalize_to_numpy
from .plot import TimestepPredictions, plot_multistep_pca
from .sample import VizSampleData, extract_sample0_viz

log = logging.getLogger(__name__)


@dataclass
class ValAccumulator:
    """Accumulator for streaming validation metrics.

    MEMORY OPTIMIZATION:
    - Metrics computed on full batch -> scalar -> discard tensors
    - PCA viz: sample 0 only -> O(T) not O(B×T)
    """

    scene_cos_raw: list[float] = field(default_factory=list)
    scene_cos_norm: list[float] = field(default_factory=list)
    cls_cos_raw: list[float] = field(default_factory=list)
    cls_cos_norm: list[float] = field(default_factory=list)
    in1k_accs: list[float] = field(default_factory=list)
    pca_predictions: list[TimestepPredictions] = field(default_factory=list)
    viz_samples: list[VizSampleData] = field(default_factory=list)
    initial_scene: np.ndarray | None = None
    initial_canvas_spatial: np.ndarray | None = None


def _log_pca(
    *,
    exp: comet_ml.CometExperiment,
    step: int,
    prefix: str,
    acc: ValAccumulator,
    full_img: np.ndarray,
    teacher_np: np.ndarray,
    boxes: list,
    names: list[str],
    canvas_grid_size: int,
    glimpse_grid_size: int,
    log_spatial_stats: bool,
    log_curves: bool,
) -> None:
    """Log PCA visualization from accumulator data."""
    assert acc.initial_scene is not None
    scenes = [vs.predicted_scene for vs in acc.viz_samples]
    glimpses = [vs.glimpse for vs in acc.viz_samples]
    canvas_spatials = [vs.canvas_spatial for vs in acc.viz_samples]

    # Extract local stream patches if available (for show_locals)
    locals_avp_raw = [vs.local_patches for vs in acc.viz_samples]
    has_locals = all(lp is not None for lp in locals_avp_raw)
    locals_avp: list[np.ndarray] | None = None
    if has_locals:
        locals_avp = [lp for lp in locals_avp_raw if lp is not None]

    fig_pca = plot_multistep_pca(
        full_img=full_img,
        teacher=teacher_np,
        scenes=scenes,
        glimpses=glimpses,
        boxes=boxes,
        names=names,
        scene_grid_size=canvas_grid_size,
        glimpse_grid_size=glimpse_grid_size,
        initial_scene=acc.initial_scene,
        locals_avp=locals_avp,
        hidden_spatials=canvas_spatials if canvas_spatials[0] is not None else None,
        initial_hidden_spatial=acc.initial_canvas_spatial,
        show_locals=has_locals,
        timestep_predictions=acc.pca_predictions if acc.pca_predictions else None,
    )
    log_figure(exp, fig_pca, f"{prefix}/pca", step)

    if log_spatial_stats and acc.viz_samples:
        target_stats = {"mean": float(np.mean(teacher_np)), "std": float(np.std(teacher_np))}
        pred_stats = {"mean": float(np.mean(scenes[-1])), "std": float(np.std(scenes[-1]))}
        exp.log_metrics(
            {
                f"{prefix}/target_spatial_mean": target_stats["mean"],
                f"{prefix}/target_spatial_std": target_stats["std"],
                f"{prefix}/pred_spatial_mean": pred_stats["mean"],
                f"{prefix}/pred_spatial_std": pred_stats["std"],
            },
            step=step,
        )

    if log_curves:
        for suffix, data in [("raw", acc.scene_cos_raw), ("norm", acc.scene_cos_norm)]:
            log_curve(
                exp,
                f"{prefix}/scene_cos_{suffix}_vs_timestep",
                x=list(range(len(data))),
                y=data,
                step=step,
            )
        if acc.cls_cos_raw:
            for suffix, data in [("raw", acc.cls_cos_raw), ("norm", acc.cls_cos_norm)]:
                log_curve(
                    exp,
                    f"{prefix}/cls_cos_{suffix}_vs_timestep",
                    x=list(range(len(data))),
                    y=data,
                    step=step,
                )


def validate(
    *,
    exp: comet_ml.CometExperiment,
    step: int,
    model: CanViTForPretraining,
    compute_raw_targets: Callable[[Tensor, int], "NormFeatures"],
    scene_normalizer: PatchStandardizer,
    cls_normalizer: CLSStandardizer,
    images: Tensor,
    canvas_grid_size: int,
    scene_size_px: int,
    glimpse_size_px: int,
    n_eval_viewpoints: int = 10,
    min_viewpoint_scale: float = 0.05,
    prefix: str = "val",
    probe: DINOv3LinearClassificationHead | None = None,
    labels: Tensor | None = None,
    log_curves: bool = False,
    log_pca: bool = False,
    teacher: DINOv3Teacher | None = None,
    log_spatial_stats: bool = False,
    teacher_name: str | None = None,
) -> float:
    """Run validation with streaming metrics (no O(B×T) memory)."""
    assert not log_pca or teacher is not None

    if probe is not None and teacher_name is not None:
        probe_res = get_probe_resolution(teacher_name)
        if scene_size_px != probe_res:
            log.warning(
                f"Resolution mismatch: model predicts teacher@{scene_size_px}, "
                f"but probe trained on teacher@{probe_res}. IN1k metrics may be unreliable."
            )

    B = images.shape[0]
    viewpoints = make_eval_viewpoints(B, images.device, n_viewpoints=n_eval_viewpoints)
    has_cls = model.scene_cls_head is not None
    has_probe = probe is not None and labels is not None and labels_are_in1k(labels)

    model_was_training = model.training
    model.eval()

    try:
        with torch.inference_mode():
            raw_feats = compute_raw_targets(images, scene_size_px)
            # Normalized targets for normalized cosine similarity and PCA
            target = scene_normalizer(raw_feats.patches)
            cls_target = cls_normalizer(raw_feats.cls.unsqueeze(1)).squeeze(1) if has_cls else None
            target_sample0 = target[0].cpu().float().numpy() if log_pca else None

            gt_idx = int(labels[0].item()) if has_probe and labels is not None else 0
            gt_name = get_imagenet_class_names()[gt_idx] if has_probe else ""

            if has_probe and teacher is not None:
                assert teacher_name is not None and probe is not None
                probe_res = get_probe_resolution(teacher_name)
                images_at_probe_res = F.interpolate(
                    images, size=(probe_res, probe_res), mode="bilinear", align_corners=False
                )
                teacher_cls = teacher.forward_norm_features(images_at_probe_res).cls
                with torch.autocast(device_type=teacher_cls.device.type, enabled=False):
                    teacher_logits = probe(teacher_cls.float())
                assert labels is not None
                teacher_acc = compute_in1k_top1(teacher_logits, labels)
                exp.log_metric(f"{prefix}/in1k_teacher_top1", teacher_acc, step=step)

            def init_fn(state: RecurrentState) -> ValAccumulator:
                acc = ValAccumulator()
                if log_pca:
                    n_canvas_tokens = model.n_canvas_registers + canvas_grid_size ** 2
                    assert_shape(state.canvas, (B, n_canvas_tokens, model.canvas_dim))
                    acc.initial_scene = (
                        model.predict_teacher_scene(state.canvas)[0].cpu().float().numpy()
                    )
                    acc.initial_canvas_spatial = (
                        model.get_spatial(state.canvas[0:1])[0].cpu().float().numpy()
                    )
                return acc

            def step_fn(
                acc: ValAccumulator, out: CanViTOutput, _vp: CanvitViewpoint, glimpse: Tensor
            ) -> ValAccumulator:
                predicted_scene = model.predict_teacher_scene(out.state.canvas)
                predicted_cls = (
                    model.predict_scene_teacher_cls(out.state.recurrent_cls) if has_cls else None
                )

                # Cosine similarity: both raw (stable across runs) and normalized
                scene_pred_raw = scene_normalizer.destandardize(predicted_scene)
                acc.scene_cos_raw.append(F.cosine_similarity(scene_pred_raw, raw_feats.patches, dim=-1).mean().item())
                acc.scene_cos_norm.append(F.cosine_similarity(predicted_scene, target, dim=-1).mean().item())

                if has_cls and predicted_cls is not None:
                    assert cls_target is not None
                    cls_pred_raw = cls_normalizer.destandardize(predicted_cls.unsqueeze(1)).squeeze(1)
                    acc.cls_cos_raw.append(F.cosine_similarity(cls_pred_raw, raw_feats.cls, dim=-1).mean().item())
                    acc.cls_cos_norm.append(F.cosine_similarity(predicted_cls, cls_target, dim=-1).mean().item())

                    if has_probe:
                        assert probe is not None and labels is not None
                        with torch.autocast(device_type=cls_pred_raw.device.type, enabled=False):
                            logits = probe(cls_pred_raw.float())
                        acc.in1k_accs.append(compute_in1k_top1(logits, labels))

                        if log_pca:
                            top_k = get_top_k_predictions(logits[0:1], k=5)[0]
                            acc.pca_predictions.append(
                                TimestepPredictions(
                                    predictions=top_k, gt_idx=gt_idx, gt_name=gt_name
                                )
                            )

                if log_pca:
                    acc.viz_samples.append(extract_sample0_viz(out, glimpse, predicted_scene, model))

                return acc

            acc, _final_state = model.forward_reduce(
                image=images,
                viewpoints=viewpoints,  # pyright: ignore[reportArgumentType]
                glimpse_size_px=glimpse_size_px,
                canvas_grid_size=canvas_grid_size,
                init_fn=init_fn,
                step_fn=step_fn,
            )

            # Log both raw and normalized cosine similarities
            exp.log_metric(f"{prefix}/scene_cos_raw", acc.scene_cos_raw[-1], step=step)
            exp.log_metric(f"{prefix}/scene_cos_norm", acc.scene_cos_norm[-1], step=step)
            for t, (raw, norm) in enumerate(zip(acc.scene_cos_raw, acc.scene_cos_norm)):
                exp.log_metric(f"{prefix}/scene_cos_raw_t{t}", raw, step=step)
                exp.log_metric(f"{prefix}/scene_cos_norm_t{t}", norm, step=step)

            if has_cls:
                exp.log_metric(f"{prefix}/cls_cos_raw", acc.cls_cos_raw[-1], step=step)
                exp.log_metric(f"{prefix}/cls_cos_norm", acc.cls_cos_norm[-1], step=step)
                for t, (raw, norm) in enumerate(zip(acc.cls_cos_raw, acc.cls_cos_norm)):
                    exp.log_metric(f"{prefix}/cls_cos_raw_t{t}", raw, step=step)
                    exp.log_metric(f"{prefix}/cls_cos_norm_t{t}", norm, step=step)

            if has_probe:
                for t, ia in enumerate(acc.in1k_accs):
                    exp.log_metric(f"{prefix}/in1k_tts_top1_t{t}", ia, step=step)
                if log_curves:
                    log_curve(
                        exp,
                        f"{prefix}/in1k_tts_top1_vs_timestep",
                        x=list(range(len(acc.in1k_accs))),
                        y=acc.in1k_accs,
                        step=step,
                    )

            if log_pca:
                assert target_sample0 is not None
                H, W = images.shape[-2], images.shape[-1]
                boxes = [vp.to_pixel_box(0, H, W) for vp in viewpoints]
                names = [vp.name for vp in viewpoints]
                full_img = imagenet_denormalize_to_numpy(images[0])
                glimpse_grid_size = glimpse_size_px // model.backbone.patch_size_px

                _log_pca(
                    exp=exp,
                    step=step,
                    prefix=prefix,
                    acc=acc,
                    full_img=full_img,
                    teacher_np=target_sample0,
                    boxes=boxes,
                    names=names,
                    canvas_grid_size=canvas_grid_size,
                    glimpse_grid_size=glimpse_grid_size,
                    log_spatial_stats=log_spatial_stats,
                    log_curves=log_curves,
                )

            return acc.scene_cos_raw[-1]
    finally:
        if model_was_training:
            model.train()
