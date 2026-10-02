"""MLX module helpers for parameters that remain outside optimization."""

from typing import Any

from mlx import nn


class Module(nn.Module):
    def trainable_parameters(self) -> dict[str, Any]:
        return self.filter_and_map(lambda module, key, value: module.trainable_parameter_filter(module, key, value))


__all__ = ["Module"]
