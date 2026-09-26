"""IN1k classification probe utilities for DINOv3 features."""

from functools import lru_cache
from typing import NamedTuple

import torch
import torch.nn.functional as F
from dinov3_in1k_probes import DINOv3LinearClassificationHead
from dinov3_in1k_probes.repos import probe_repo
from torch import Tensor
from torchvision.models import ResNet50_Weights

IN1K_NUM_CLASSES = 1000


class ProbeInfo(NamedTuple):
    """Probe metadata tied to its training configuration."""
    repo: str
    resolution: int  # Image resolution the probe was trained at (pixels)


# All probes trained at 512x512.
PROBE_REGISTRY: dict[str, ProbeInfo] = {
    "dinov3_vits16": ProbeInfo(probe_repo("vits16"), 512),
    "dinov3_vitb16": ProbeInfo(probe_repo("vitb16"), 512),
    "dinov3_vitl16": ProbeInfo(probe_repo("vitl16"), 512),
}


class TopKPrediction(NamedTuple):
    """A single top-k prediction with class info and probability."""
    class_idx: int
    class_name: str
    probability: float


@lru_cache(maxsize=1)
def get_imagenet_class_names() -> list[str]:
    """Get ImageNet-1k class names from torchvision."""
    return list(ResNet50_Weights.IMAGENET1K_V1.meta["categories"])


def load_probe(teacher_name: str, device: torch.device) -> DINOv3LinearClassificationHead | None:
    """Load IN1k classification probe from HF Hub. Returns None if teacher_name unsupported."""
    if teacher_name not in PROBE_REGISTRY:
        return None
    probe = DINOv3LinearClassificationHead.from_pretrained(PROBE_REGISTRY[teacher_name].repo)
    return probe.to(device).eval()


def get_probe_resolution(teacher_name: str) -> int:
    """Get the image resolution the probe was trained at. Raises KeyError if teacher_name unsupported."""
    return PROBE_REGISTRY[teacher_name].resolution


def labels_are_in1k(labels: Tensor) -> bool:
    """Check if labels are from IN1k (< 1000) vs IN21k (>= 1000)."""
    return int(labels.max().item()) < IN1K_NUM_CLASSES


def compute_in1k_top1(logits: Tensor, labels: Tensor) -> float:
    """Compute top-1 accuracy as percentage.

    Caller must ensure labels are IN1k (use labels_are_in1k() to check).
    """
    assert labels_are_in1k(labels), f"Labels {labels.max().item()} exceed IN1k range"
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).sum().item()
    return 100.0 * correct / labels.shape[0]


def get_top_k_predictions(logits: Tensor, k: int = 5) -> list[list[TopKPrediction]]:
    """Top-k predictions with class names and probabilities from [B, num_classes] logits."""
    class_names = get_imagenet_class_names()
    probs = F.softmax(logits, dim=-1)
    top_probs, top_indices = probs.topk(k, dim=-1)

    results: list[list[TopKPrediction]] = []
    for b in range(logits.shape[0]):
        preds = []
        for i in range(k):
            idx = top_indices[b, i].item()
            assert isinstance(idx, int)
            preds.append(TopKPrediction(
                class_idx=idx,
                class_name=class_names[idx],
                probability=top_probs[b, i].item(),
            ))
        results.append(preds)
    return results
