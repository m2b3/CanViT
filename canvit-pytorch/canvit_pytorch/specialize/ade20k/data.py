"""ADE20K for probe training: augmented training scenes, and the validation split as the paper evaluates it."""

import itertools
from collections.abc import Iterator

import torch
from dinov3.eval.segmentation.transforms import make_segmentation_train_transforms
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader

from canvit_pytorch.benchmarks.ade20k import IGNORE_LABEL, ADE20kDataset, dataset_root, evaluation_transform
from canvit_pytorch.specialize.ade20k.config import ProbeTrainingConfig

assert IGNORE_LABEL == 255, "DINOv3's training transforms mark ignored and padded pixels with 255"


class TrainingAugmentation:
    """DINOv3's segmentation training augmentation: scale jitter, a random square crop (padded when the image is
    smaller), horizontal flip, photometric distortion. ADE20K's label 0 ("other") becomes IGNORE_LABEL."""

    def __init__(self, config: ProbeTrainingConfig) -> None:
        self.augment = make_segmentation_train_transforms(
            img_size=config.scene_size_px,
            random_img_size_ratio_range=list(config.scale_jitter_range),
            # Annotated Tuple[int] upstream, read as (height, width).
            crop_size=(config.scene_size_px, config.scene_size_px),  # pyright: ignore[reportArgumentType]
            flip_prob=config.horizontal_flip_probability,
            reduce_zero_label=True,
        )

    def __call__(self, image: Image.Image, label_map: Image.Image) -> tuple[Tensor, Tensor]:
        image_tensor, labels = self.augment(image, label_map)
        return image_tensor, labels.squeeze(0).long()


def training_batches(config: ProbeTrainingConfig) -> Iterator[tuple[Tensor, Tensor]]:
    """Batches of augmented training scenes [B, 3, S, S] and labels [B, S, S], shuffled, epoch after epoch."""
    dataset = ADE20kDataset(root=dataset_root(), split="training", transform=TrainingAugmentation(config))
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, drop_last=True,
        num_workers=config.num_workers, pin_memory=_pin_memory(config),
    )
    return itertools.chain.from_iterable(itertools.repeat(loader))


def validation_loader(config: ProbeTrainingConfig) -> DataLoader[tuple[Tensor, Tensor]]:
    dataset = ADE20kDataset(
        root=dataset_root(), split="validation", transform=evaluation_transform(config.scene_size_px),
    )
    return DataLoader(
        dataset, batch_size=config.validation_batch_size,
        num_workers=config.num_workers, pin_memory=_pin_memory(config),
    )


def _pin_memory(config: ProbeTrainingConfig) -> bool:
    return torch.device(config.device).type == "cuda"
