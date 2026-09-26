"""Segmentation probe utilities for ADE20K evaluation."""

import torch.nn as nn
from torch import Tensor

from canvit_specialize.metrics import mIoUAccumulator
from canvit_specialize.training.ade20k.loss import upsample_preds


def eval_probe_on_batch(
    probe: nn.Module,
    features: Tensor,
    masks: Tensor,
    iou: mIoUAccumulator,
) -> None:
    """Forward probe, upsample predictions, update IoU accumulator."""
    logits = probe(features.float())
    preds_up = upsample_preds(logits.argmax(1), masks.shape[1], masks.shape[2])
    iou.update(preds_up, masks)
