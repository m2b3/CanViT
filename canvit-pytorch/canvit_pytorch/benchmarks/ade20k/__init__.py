"""ADE20K SceneParsing, the paper's dense-prediction benchmark."""

from canvit_pytorch.benchmarks.ade20k.classes import CLASS_NAMES, IGNORE_LABEL, NUM_CLASSES
from canvit_pytorch.benchmarks.ade20k.dataset import (
    NUM_VALIDATION_IMAGES,
    ADE20kDataset,
    dataset_root,
    decode_annotation,
    evaluation_transform,
)
from canvit_pytorch.benchmarks.ade20k.metrics import MeanIoU, per_image_confusion

__all__ = [
    "CLASS_NAMES", "IGNORE_LABEL", "NUM_CLASSES", "NUM_VALIDATION_IMAGES",
    "ADE20kDataset", "MeanIoU", "dataset_root", "decode_annotation", "evaluation_transform", "per_image_confusion",
]
