"""Loss functions for ADE20K probe training."""

import torch.nn.functional as F
from torch import Tensor

from canvit_specialize.datasets.ade20k import IGNORE_LABEL


def ce_loss(logits: Tensor, masks: Tensor) -> Tensor:
    """Cross-entropy loss for semantic segmentation."""
    if masks.shape[1:] != logits.shape[2:]:
        masks = F.interpolate(masks.unsqueeze(1).float(), logits.shape[2:], mode="nearest").squeeze(1).long()
    return F.cross_entropy(logits, masks, ignore_index=IGNORE_LABEL)


def upsample_preds(preds: Tensor, H: int, W: int) -> Tensor:
    if preds.shape[1:] == (H, W):
        return preds
    return F.interpolate(preds.unsqueeze(1).float(), (H, W), mode="nearest").squeeze(1).long()
