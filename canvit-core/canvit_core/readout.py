"""Fusing a linear probe fitted on the teacher's CLS token into CanViT's CLS readout."""

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
    """Collapse three affine maps into one: project to teacher space, destandardize, apply the teacher's probe.

        logits = W_probe (std ⊙ (W_proj z + b_proj) + mean) + b_probe = W_fused z + b_fused

    Computed in float64; returns (W_fused [n_classes, D], b_fused [n_classes]) in proj_weight's dtype.
    """
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
