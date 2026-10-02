import numpy as np
import pytest

from tools.checkpoint_conversion import CheckpointSchemaError, convert_named_arrays
from tools.convert_checkpoints import _assert_metrics


def test_convert_named_arrays_transposes_mlx_convolution():
    source = {"backbone.patch_embed.proj.weight": np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)}
    converted = convert_named_arrays(
        source,
        target_shapes={"backbone.patch_embed.proj.weight": (2, 4, 5, 3)},
        backend="mlx",
    )
    np.testing.assert_array_equal(converted["backbone.patch_embed.proj.weight"], source[next(iter(source))].transpose(0, 2, 3, 1))


def test_convert_named_arrays_maps_nnx_parameter_names_and_shapes():
    source = {
        "linear.weight": np.arange(6, dtype=np.float32).reshape(3, 2),
        "norm.weight": np.arange(3, dtype=np.float32),
        "norm.bias": np.arange(3, dtype=np.float32) + 1,
    }
    converted = convert_named_arrays(
        source,
        target_shapes={"linear.kernel": (2, 3), "norm.scale": (3,), "norm.bias": (3,)},
        backend="nnx",
    )
    np.testing.assert_array_equal(converted["linear.kernel"], source["linear.weight"].T)
    np.testing.assert_array_equal(converted["norm.scale"], source["norm.weight"])
    np.testing.assert_array_equal(converted["norm.bias"], source["norm.bias"])


def test_convert_named_arrays_transposes_nnx_convolution():
    source = {"patch.proj.weight": np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)}
    converted = convert_named_arrays(
        source,
        target_shapes={"patch.proj.kernel": (4, 5, 3, 2)},
        backend="nnx",
    )
    np.testing.assert_array_equal(converted["patch.proj.kernel"], source["patch.proj.weight"].transpose(2, 3, 1, 0))


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ({"a": np.ones(2)}, {"b": (2,)}),
        ({"a": np.ones(2), "b": np.ones(2)}, {"a": (2,)}),
        ({"a": np.ones(2)}, {"a": (3,)}),
    ],
)
def test_convert_named_arrays_refuses_schema_mismatches(source, target):
    with pytest.raises(CheckpointSchemaError):
        convert_named_arrays(source, target_shapes=target, backend="mlx")


def test_parity_metrics_reject_nonfinite_values():
    with pytest.raises(FloatingPointError):
        _assert_metrics({
            "step_0.canvas": {
                "max_abs": float("nan"),
                "relative_l2": 0.0,
                "normalized_l2": 0.0,
                "normalized_rmse": 0.0,
            },
        })
