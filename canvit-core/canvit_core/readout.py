import numpy as np
from numpy.typing import NDArray


def fuse_probe(
    *,
    proj_weight: NDArray,
    proj_bias: NDArray,
    mean: NDArray,
    std: NDArray,
    probe_weight: NDArray,
    probe_bias: NDArray,
) -> tuple[NDArray, NDArray]:
    teacher_dim = proj_weight.shape[0]
    assert mean.shape == std.shape == proj_bias.shape == (teacher_dim,)
    assert probe_weight.shape == (probe_bias.shape[0], teacher_dim)
    dtype = proj_weight.dtype
    proj_weight, proj_bias, mean, std, probe_weight, probe_bias = (
        value.astype(np.float64) for value in (proj_weight, proj_bias, mean, std, probe_weight, probe_bias)
    )
    return (
        (probe_weight @ (std[:, None] * proj_weight)).astype(dtype),
        (probe_weight @ (std * proj_bias + mean) + probe_bias).astype(dtype),
    )
