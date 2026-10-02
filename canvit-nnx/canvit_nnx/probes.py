"""Linear probes for native NHWC spatial feature maps."""

from typing import Any, Self

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from canvit_nnx.hub import HubMixin

Array = jax.Array


class _BatchNorm2d(nnx.Module):
    def __init__(self, *, num_features: int, momentum: float, epsilon: float) -> None:
        self.weight = nnx.Param(jnp.ones((num_features,), dtype=jnp.float32))
        self.bias = nnx.Param(jnp.zeros((num_features,), dtype=jnp.float32))
        self.running_mean = nnx.BatchStat(jnp.zeros((num_features,), dtype=jnp.float32))
        self.running_var = nnx.BatchStat(jnp.ones((num_features,), dtype=jnp.float32))
        self.num_batches_tracked = nnx.BatchStat(jnp.asarray(0, dtype=jnp.uint32))
        self.momentum = momentum
        self.epsilon = epsilon
        self.use_running_average = False

    def __call__(self, x: Array) -> Array:
        assert x.ndim == 4, x.shape
        sample_count = x.shape[0] * x.shape[1] * x.shape[2]
        if self.use_running_average:
            mean = self.running_mean[...]
            variance = self.running_var[...]
        else:
            assert sample_count > 1, "BatchNorm training requires more than one value per channel"
            mean = jnp.mean(x, axis=(0, 1, 2))
            centered = x - mean
            variance = jnp.mean(centered * centered, axis=(0, 1, 2))
            # PyTorch stores the unbiased batch variance in running_var while normalizing with the biased variance.
            unbiased_variance = variance * (sample_count / (sample_count - 1))
            self.running_mean[...] = (1.0 - self.momentum) * self.running_mean[...] + self.momentum * mean
            self.running_var[...] = (1.0 - self.momentum) * self.running_var[...] + self.momentum * unbiased_variance
            self.num_batches_tracked[...] = self.num_batches_tracked[...] + 1
        normalized = (x - mean) * jax.lax.rsqrt(variance + self.epsilon)
        return normalized * self.weight[...] + self.bias[...]


class SegmentationProbe(nnx.Module, HubMixin):
    def __init__(
        self,
        *,
        embed_dim: int,
        num_classes: int,
        dropout: float,
        use_ln: bool,
        rngs: nnx.Rngs,
    ) -> None:
        assert embed_dim > 0 and num_classes > 0, (embed_dim, num_classes)
        assert 0.0 <= dropout < 1.0, dropout
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.dropout_probability = dropout
        self.use_ln = use_ln
        if use_ln:
            self.ln = nnx.LayerNorm(
                embed_dim,
                epsilon=1e-5,
                use_fast_variance=False,
                rngs=rngs,
            )
        self.bn = _BatchNorm2d(num_features=embed_dim, momentum=0.1, epsilon=1e-5)
        self.dropout = nnx.Dropout(dropout, broadcast_dims=(1, 2), rngs=None)
        self.conv = nnx.Conv(
            in_features=embed_dim,
            out_features=num_classes,
            kernel_size=(1, 1),
            kernel_init=jax.nn.initializers.normal(stddev=0.01),
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs,
        )

    def __call__(self, features: Array, *, rngs: jax.Array | nnx.Rngs | None = None) -> Array:
        """Map [B, H, W, embed_dim] features to [B, H, W, num_classes] logits."""
        assert features.ndim == 4, features.shape
        assert features.shape[-1] == self.embed_dim, (
            f"probe expects {self.embed_dim}-dim features, got {features.shape[-1]}"
        )
        x = self.ln(features) if self.use_ln else features
        x = self.dropout(x, rngs=rngs)
        return self.conv(self.bn(x))

    def checkpoint_config(self) -> dict[str, Any]:
        return {
            "embed_dim": self.embed_dim,
            "num_classes": self.num_classes,
            "dropout": self.dropout_probability,
            "use_ln": self.use_ln,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        return cls(**config, rngs=nnx.Rngs(0))


__all__ = ["SegmentationProbe"]
