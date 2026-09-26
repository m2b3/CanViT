"""CanViT pretraining training loop: data, probe, scheduler, viewpoint, viz."""

from canvit_pretrain.train.data import (
    Batch,
    InfiniteLoader,
    Loaders,
    create_loaders,
    scene_size_px,
)
from canvit_pretrain.train.probe import (
    IN1K_NUM_CLASSES,
    PROBE_REGISTRY,
    ProbeInfo,
    TopKPrediction,
    compute_in1k_top1,
    get_imagenet_class_names,
    get_probe_resolution,
    get_top_k_predictions,
    labels_are_in1k,
    load_probe,
)
from canvit_pretrain.train.scheduler import warmup_constant_scheduler
from canvit_pretrain.train.viewpoint import (
    PixelBox,
    Viewpoint,
    make_eval_viewpoints,
    viewpoint_to_pixel_box,
)
from canvit_pretrain.train.viz import (
    TimestepPredictions,
    fit_pca,
    imagenet_denormalize_to_numpy,
    pca_rgb,
    plot_multistep_pca,
    plot_pca_grid,
    plot_trajectory,
    timestep_colors,
    validate,
)

__all__ = [
    # Data
    "Batch",
    "InfiniteLoader",
    "Loaders",
    "create_loaders",
    "scene_size_px",
    # Probe
    "IN1K_NUM_CLASSES",
    "PROBE_REGISTRY",
    "ProbeInfo",
    "TopKPrediction",
    "compute_in1k_top1",
    "get_imagenet_class_names",
    "get_probe_resolution",
    "get_top_k_predictions",
    "labels_are_in1k",
    "load_probe",
    # Scheduler
    "warmup_constant_scheduler",
    # Viewpoint
    "PixelBox",
    "Viewpoint",
    "make_eval_viewpoints",
    "viewpoint_to_pixel_box",
    # Viz
    "TimestepPredictions",
    "fit_pca",
    "imagenet_denormalize_to_numpy",
    "pca_rgb",
    "plot_multistep_pca",
    "plot_pca_grid",
    "plot_trajectory",
    "timestep_colors",
    "validate",
]
