import math

import mlx.core as mx
from mlx import nn

from canvit_mlx.module import Module
from canvit_mlx.viewpoint import Viewpoint


class ViewpointEncoding(Module):
    def __init__(self, dim: int, seed: int = 42) -> None:
        super().__init__()
        self.frequencies = mx.random.normal((dim // 2, 3), key=mx.random.key(seed))
        self.norm = nn.LayerNorm(dim)
        self.norm.weight = mx.full((dim,), 1 / math.sqrt(dim))
        self.freeze(keys=["frequencies"], recurse=False, strict=True)

    @staticmethod
    def trainable_parameter_filter(module: nn.Module, key: str, value: object) -> bool:
        return Module.trainable_parameter_filter(module, key, value) and key != "frequencies"

    def __call__(self, viewpoint: Viewpoint) -> mx.array:
        coordinates = mx.concatenate(
            [
                viewpoint.centers / viewpoint.scales[:, None],
                mx.log(viewpoint.scales[:, None]),
            ],
            axis=-1,
        )
        phases = coordinates @ self.frequencies.T
        return self.norm(mx.concatenate([mx.cos(phases), mx.sin(phases)], axis=-1))
