"""Shared constants, TFRecord dataloaders, and classifier loader for TPU training."""

import io
import logging
from pathlib import Path

log = logging.getLogger("shared")

import numpy as np
import torch
import torchvision.transforms.v2 as T

from canvit_pytorch.checkpoints import FLAGSHIP_PRETRAIN_REPO
from dinov3_in1k_probes.repos import probe_repo
from PIL import Image
from tfrecord.reader import tfrecord_loader

CKPT = FLAGSHIP_PRETRAIN_REPO

SCENE_SIZE = 512
GLIMPSE_SIZE = 128
CANVAS_GRID = 32

from canvit_pytorch.preprocess import IMAGENET_DEFAULT_MEAN as IMAGENET_MEAN, IMAGENET_DEFAULT_STD as IMAGENET_STD

TFRECORD_DESCRIPTION = {
    "image/encoded": "byte",
    "image/class/label": "int",
}

TRAIN_TRANSFORM = T.Compose([
    T.RandomResizedCrop(SCENE_SIZE, scale=(0.2, 1.0)),  # BILINEAR (default) — matches pretraining
    T.RandomHorizontalFlip(),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

VAL_TRANSFORM = T.Compose([
    T.Resize(SCENE_SIZE),  # BILINEAR (default) — matches pretraining
    T.CenterCrop(SCENE_SIZE),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def decode_tfrecord(features: dict, transform=None) -> tuple[torch.Tensor, int]:
    """Decode a TFRecord into (image_tensor, label). Labels are 0-indexed."""
    if transform is None:
        transform = VAL_TRANSFORM
    jpeg_bytes = features["image/encoded"]
    label = int(features["image/class/label"].item()) - 1  # 1-indexed → 0-indexed
    label = max(0, label)
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    return transform(img), label


class ShardedTFRecordDataset(torch.utils.data.IterableDataset):
    """Iterate over TFRecord shards with shuffling. No index files needed."""

    def __init__(self, shards: list[Path], transform=None):
        assert len(shards) > 0, "No shards provided"
        self.shards = shards
        self.transform = transform or VAL_TRANSFORM

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        shards = self.shards
        if worker_info is not None:
            shards = shards[worker_info.id::worker_info.num_workers]
        rng = np.random.default_rng()
        while True:
            for idx in rng.permutation(len(shards)):
                for record in tfrecord_loader(str(shards[idx]), None, TFRECORD_DESCRIPTION):
                    try:
                        yield decode_tfrecord(record, transform=self.transform)
                    except Exception:
                        continue


def find_shards(data_dir: str, split: str = "train") -> list[Path]:
    """Find TFRecord shards for a split ('train' or 'validation')."""
    shards = sorted(p for p in Path(data_dir).glob(f"{split}-*") if not p.suffix)
    assert len(shards) > 0, f"No {split} shards found in {data_dir}"
    log.info("Found %d %s shards in %s", len(shards), split, data_dir)
    return shards


from typing import Literal

from canvit_pytorch.policies import coarse_to_fine_viewpoints

ViewpointPolicy = Literal["random", "c2f"]


def c2f_viewpoints(
    batch_size: int,
    device: torch.device,
    n_viewpoints: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Coarse-to-Fine quadtree viewpoints as (centers, scales) tuples.

    Thin wrapper around canvit_pytorch.policies.coarse_to_fine_viewpoints.
    """
    vps = coarse_to_fine_viewpoints(batch_size, device, n_viewpoints)
    return [(vp.centers, vp.scales) for vp in vps]


def make_multi_glimpse_dataloader(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    n_glimpses: int,
    split: str = "train",
    glimpse_size: int = GLIMPSE_SIZE,
    min_viewpoint_scale: float = 0.05,
    t0_full_scene: bool = True,
    viewpoint_policy: ViewpointPolicy = "random",
) -> torch.utils.data.DataLoader:
    """DataLoader for multi-glimpse training/eval. All N glimpses sampled in CPU workers.

    viewpoint_policy:
    - "random": t=0 full_scene (if t0_full_scene), t>=1 random with p(s) ∝ (1-s).
    - "c2f": Coarse-to-Fine quadtree (full scene → quadrants → sub-quadrants).
      Within-level order shuffled per batch element. See c2f_viewpoints().

    Returns per batch: (glimpses [N,B,3,G,G], labels [B], vp_centers [N,B,2], vp_scales [N,B])
    """
    from canvit_pytorch import Viewpoint, sample_at_viewpoint

    transform = TRAIN_TRANSFORM if split == "train" else VAL_TRANSFORM
    shards = find_shards(data_dir, split=split)
    dataset = ShardedTFRecordDataset(shards, transform=transform)

    def _random_vp(b: int) -> Viewpoint:
        L_max = 1 - min_viewpoint_scale
        u = torch.rand(b)
        L = torch.sqrt(u * (L_max**2))
        scales = 1 - L
        centers = (torch.rand(b, 2) * 2 - 1) * L.unsqueeze(1)
        return Viewpoint(centers=centers, scales=scales)

    def collate_multi_glimpse(batch):
        images = torch.stack([x[0] for x in batch])
        labels = torch.tensor([x[1] for x in batch], dtype=torch.long)
        b = images.shape[0]

        if viewpoint_policy == "c2f":
            vps = c2f_viewpoints(b, torch.device("cpu"), n_glimpses)
        else:
            vps = None  # generate per-step below

        all_glimpses, all_centers, all_scales = [], [], []
        for g in range(n_glimpses):
            if vps is not None:
                centers, scales = vps[g]
                vp = Viewpoint(centers=centers, scales=scales)
            elif g == 0 and t0_full_scene:
                vp = Viewpoint.full_scene(batch_size=b, device="cpu")
            else:
                vp = _random_vp(b)
            glimpse = sample_at_viewpoint(spatial=images, viewpoint=vp, glimpse_size_px=glimpse_size)
            all_glimpses.append(glimpse)
            all_centers.append(vp.centers)
            all_scales.append(vp.scales)

        return torch.stack(all_glimpses), labels, torch.stack(all_centers), torch.stack(all_scales)

    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True,
        collate_fn=collate_multi_glimpse,
        prefetch_factor=2 if num_workers > 0 else None,
    )


from canvit_pytorch import CanViTForImageClassification

PROBE_REPO = probe_repo("vitb16")


def load_classifier(device: torch.device) -> CanViTForImageClassification:
    """Load pretrained CanViT with fused DINOv3 probe head."""
    import time as _time
    t0 = _time.perf_counter()
    clf = CanViTForImageClassification.from_pretrained_with_probe(
        pretrained_repo=CKPT, probe_repo=PROBE_REPO, canvas_grid=CANVAS_GRID,
    ).to(device)
    log.info("Loaded classifier: %s params, %.1fs",
             f"{sum(p.numel() for p in clf.parameters()):,}", _time.perf_counter() - t0)
    return clf
