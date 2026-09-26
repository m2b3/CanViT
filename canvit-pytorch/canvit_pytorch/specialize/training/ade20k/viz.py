"""Visualization for ADE20K probe training (canvas and single-probe)."""

from collections.abc import Mapping
from typing import Protocol

import comet_ml
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from torch import Tensor

from canvit_specialize.datasets.ade20k import IGNORE_LABEL, NUM_CLASSES
from canvit_specialize.training.ade20k.config import CanvasFeatureType
from canvit_specialize.training.ade20k.features import CanvasFeatures
from canvit_pytorch.preprocess import imagenet_denormalize


class ProbeStateLike(Protocol):
    @property
    def head(self) -> torch.nn.Module: ...

# Deterministic palette for segmentation masks
_PALETTE = np.random.RandomState(42).randint(0, 255, (NUM_CLASSES + 1, 3), dtype=np.uint8)
_PALETTE[NUM_CLASSES] = 0


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    return _PALETTE[np.where(mask == IGNORE_LABEL, NUM_CLASSES, mask)]


def _fit_pca(feats: np.ndarray, n_components: int = 3) -> PCA | None:
    if feats.var(axis=0).max() < 1e-5:
        return None
    n_components = min(n_components, feats.shape[0], feats.shape[1])
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(feats)
    return pca


def _pca_to_rgb(pca: PCA | None, feats: np.ndarray, H: int, W: int) -> np.ndarray:
    if pca is None:
        return np.full((H, W, 3), 0.5, dtype=np.float32)
    proj = pca.transform(feats)[:, :3]
    lo = np.percentile(proj, 2, axis=0, keepdims=True)
    hi = np.percentile(proj, 98, axis=0, keepdims=True)
    rgb = np.clip((proj - lo) / (hi - lo + 1e-8), 0, 1)
    return rgb.reshape(H, W, 3).astype(np.float32)


def correctness_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    H, W = pred.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    valid = gt != IGNORE_LABEL
    out[(pred == gt) & valid] = [0, 200, 0]
    out[(pred != gt) & valid] = [200, 0, 0]
    out[~valid] = [128, 128, 128]
    return out


def make_viz_figure(
    probes: Mapping[CanvasFeatureType, ProbeStateLike],
    feats: CanvasFeatures,
    images: Tensor,
    masks: Tensor,
    n_samples: int,
    n_timesteps: int,
) -> Figure:
    n_samples = min(n_samples, images.shape[0])
    feat_types = list(probes.keys())
    n_cols = 2 + len(feat_types) * 6
    t_final = n_timesteps - 1

    fig, axes = plt.subplots(n_samples, n_cols, figsize=(2.5 * n_cols, 2.5 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_samples):
        col = 0
        # imagenet_denormalize returns CHW tensor; matplotlib imshow requires HWC.
        img_np = (imagenet_denormalize(images[i]).cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        axes[i, col].imshow(img_np)
        axes[i, col].set_title("Image" if i == 0 else "")
        axes[i, col].axis("off")
        col += 1

        gt = masks[i].cpu().numpy()
        axes[i, col].imshow(colorize_mask(gt))
        axes[i, col].set_title("GT" if i == 0 else "")
        axes[i, col].axis("off")
        col += 1

        pca_per_feat: dict[CanvasFeatureType, PCA | None] = {}
        for feat_type in feat_types:
            feat_final = feats.get(feat_type, t_final)[i].cpu().float().numpy()
            H, W, D = feat_final.shape
            pca_per_feat[feat_type] = _fit_pca(feat_final.reshape(-1, D))

        for feat_type in feat_types:
            pca = pca_per_feat[feat_type]
            for t, t_name in [(0, "t0"), (t_final, "t-1")]:
                feat_i = feats.get(feat_type, t)[i]
                H, W, D = feat_i.shape

                with torch.no_grad():
                    logits = probes[feat_type].head(feat_i.unsqueeze(0).float())
                    pred = logits[0].argmax(0).cpu().numpy()

                pred_up = F.interpolate(
                    torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float(),
                    size=gt.shape, mode="nearest",
                ).squeeze().numpy().astype(np.int64)

                axes[i, col].imshow(colorize_mask(pred_up))
                axes[i, col].set_title(f"{feat_type[:8]} {t_name}" if i == 0 else "")
                axes[i, col].axis("off")
                col += 1

                axes[i, col].imshow(correctness_map(pred_up, gt))
                axes[i, col].set_title(f"corr {t_name}" if i == 0 else "")
                axes[i, col].axis("off")
                col += 1

                feat_np = feat_i.cpu().float().numpy()
                pca_img = _pca_to_rgb(pca, feat_np.reshape(-1, D), H, W)
                pca_up = F.interpolate(
                    torch.from_numpy(pca_img).permute(2, 0, 1).unsqueeze(0),
                    size=gt.shape, mode="bilinear", align_corners=False,
                ).squeeze().permute(1, 2, 0).numpy()
                axes[i, col].imshow(pca_up)
                axes[i, col].set_title(f"PCA {t_name}" if i == 0 else "")
                axes[i, col].axis("off")
                col += 1

    plt.tight_layout()
    return fig


def log_viz(
    exp: comet_ml.Experiment,
    step: int,
    probes: Mapping[CanvasFeatureType, ProbeStateLike],
    feats: CanvasFeatures,
    images: Tensor,
    masks: Tensor,
    n_samples: int,
    n_timesteps: int,
    split: str = "train",
) -> None:
    fig = make_viz_figure(probes, feats, images, masks, n_samples, n_timesteps)
    exp.log_figure(figure_name=f"viz_{split}_{step}", figure=fig, step=step)
    plt.close(fig)


def make_probe_viz_figure(
    probe: torch.nn.Module,
    features: Tensor,
    images: Tensor,
    masks: Tensor,
    n_samples: int,
) -> Figure:
    """Single-probe viz: [Image | GT | Pred | Correctness | PCA] per sample."""
    n_samples = min(n_samples, images.shape[0])
    n_cols = 5

    fig, axes = plt.subplots(n_samples, n_cols, figsize=(2.5 * n_cols, 2.5 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_samples):
        col = 0
        # imagenet_denormalize returns CHW tensor; matplotlib imshow requires HWC.
        img_np = (imagenet_denormalize(images[i]).cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        axes[i, col].imshow(img_np)
        axes[i, col].set_title("Image" if i == 0 else "")
        axes[i, col].axis("off")
        col += 1

        gt = masks[i].cpu().numpy()
        axes[i, col].imshow(colorize_mask(gt))
        axes[i, col].set_title("GT" if i == 0 else "")
        axes[i, col].axis("off")
        col += 1

        feat_i = features[i]
        H, W, D = feat_i.shape
        with torch.no_grad():
            logits = probe(feat_i.unsqueeze(0).float())
            pred = logits[0].argmax(0).cpu().numpy()
        pred_up = F.interpolate(
            torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float(),
            size=gt.shape, mode="nearest",
        ).squeeze().numpy().astype(np.int64)

        axes[i, col].imshow(colorize_mask(pred_up))
        axes[i, col].set_title("Pred" if i == 0 else "")
        axes[i, col].axis("off")
        col += 1

        axes[i, col].imshow(correctness_map(pred_up, gt))
        axes[i, col].set_title("Correct" if i == 0 else "")
        axes[i, col].axis("off")
        col += 1

        feat_np = feat_i.cpu().float().numpy()
        pca = _fit_pca(feat_np.reshape(-1, D))
        pca_img = _pca_to_rgb(pca, feat_np.reshape(-1, D), H, W)
        pca_up = F.interpolate(
            torch.from_numpy(pca_img).permute(2, 0, 1).unsqueeze(0),
            size=gt.shape, mode="bilinear", align_corners=False,
        ).squeeze().permute(1, 2, 0).numpy()
        axes[i, col].imshow(pca_up)
        axes[i, col].set_title("PCA" if i == 0 else "")
        axes[i, col].axis("off")
        col += 1

    plt.tight_layout()
    return fig


def log_probe_viz(
    exp: comet_ml.Experiment,
    step: int,
    probe: torch.nn.Module,
    features: Tensor,
    images: Tensor,
    masks: Tensor,
    n_samples: int,
    split: str = "val",
) -> None:
    fig = make_probe_viz_figure(probe, features, images, masks, n_samples)
    exp.log_figure(figure_name=f"viz_{split}_{step}", figure=fig, step=step)
    plt.close(fig)
