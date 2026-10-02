"""Linear probes for native NHWC spatial feature maps."""

from typing import Any, Self

import mlx.core as mx
from mlx import nn

from canvit_mlx.hub import HubMixin
from canvit_mlx.module import Module


class _ChannelDropout(nn.Module):
    def __init__(self, probability: float) -> None:
        super().__init__()
        assert 0.0 <= probability < 1.0, probability
        self.probability = probability

    def __call__(self, x: mx.array) -> mx.array:
        if self.probability == 0.0 or not self.training:
            return x
        keep_probability = 1.0 - self.probability
        mask = mx.random.bernoulli(
            keep_probability,
            shape=(x.shape[0], 1, 1, x.shape[-1]),
        )
        return mask * x / keep_probability


class _BatchNorm2d(nn.BatchNorm):
    def __init__(self, num_features: int) -> None:
        super().__init__(num_features, eps=1e-5, momentum=0.1)
        self.num_batches_tracked = mx.array(0, dtype=mx.int64)
        self.freeze(keys=["num_batches_tracked"], recurse=False)

    @staticmethod
    def trainable_parameter_filter(module: nn.Module, key: str, value: object) -> bool:
        return nn.Module.trainable_parameter_filter(module, key, value) and key not in {
            "running_mean",
            "running_var",
            "num_batches_tracked",
        }

    def __call__(self, x: mx.array) -> mx.array:
        output = super().__call__(x)
        if self.training:
            self.num_batches_tracked = self.num_batches_tracked + 1
        return output


class SegmentationProbe(Module, HubMixin):
    def __init__(self, *, embed_dim: int, num_classes: int, dropout: float, use_ln: bool) -> None:
        super().__init__()
        assert embed_dim > 0 and num_classes > 0, (embed_dim, num_classes)
        assert 0.0 <= dropout < 1.0, dropout
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.dropout_probability = dropout
        self.use_ln = use_ln
        if use_ln:
            self.ln = nn.LayerNorm(embed_dim, eps=1e-5)
        self.bn = _BatchNorm2d(embed_dim)
        self.dropout = _ChannelDropout(dropout)
        self.conv = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        self.conv.weight = mx.random.normal(shape=self.conv.weight.shape) * 0.01
        self.conv.bias = mx.zeros_like(self.conv.bias)

    def __call__(self, features: mx.array) -> mx.array:
        """Map [B, H, W, embed_dim] features to [B, H, W, num_classes] logits."""
        assert features.ndim == 4, features.shape
        assert features.shape[-1] == self.embed_dim, (
            f"probe expects {self.embed_dim}-dim features, got {features.shape[-1]}"
        )
        x = self.ln(features) if self.use_ln else features
        return self.conv(self.bn(self.dropout(x)))

    def checkpoint_config(self) -> dict[str, Any]:
        return {
            "embed_dim": self.embed_dim,
            "num_classes": self.num_classes,
            "dropout": self.dropout_probability,
            "use_ln": self.use_ln,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        return cls(**config)


__all__ = ["SegmentationProbe"]
