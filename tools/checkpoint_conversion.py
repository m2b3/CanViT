from collections.abc import Mapping
from typing import Any, Literal, cast

import numpy as np

BackendName = Literal["mlx", "nnx"]


class CheckpointSchemaError(ValueError):
    pass


def native_state_arrays(model: object, *, backend: BackendName, trainable_only: bool = False) -> dict[str, np.ndarray]:
    if backend == "mlx":
        import mlx.nn as nn
        from mlx.utils import tree_flatten

        assert isinstance(model, nn.Module)
        tree = model.trainable_parameters() if trainable_only else model.parameters()
        return {name: np.asarray(value) for name, value in tree_flatten(tree)}

    from flax import nnx

    state = nnx.state(model, nnx.Param) if trainable_only else nnx.state(model)
    flat_state = nnx.to_flat_state(state)
    return {".".join(map(str, path)): np.asarray(value[...]) for path, value in zip(flat_state.paths, flat_state.leaves)}


def native_gradient_arrays(tree: object, *, backend: BackendName) -> dict[str, np.ndarray]:
    if backend == "mlx":
        from mlx.utils import tree_flatten

        return {name: np.asarray(value) for name, value in tree_flatten(tree)}
    from flax import nnx

    flat_state = nnx.to_flat_state(cast(Any, tree))
    return {".".join(map(str, path)): np.asarray(value[...]) for path, value in zip(flat_state.paths, flat_state.leaves)}


def _source_name(target_name: str, *, backend: BackendName) -> str:
    if backend == "nnx" and target_name.endswith((".kernel", ".scale")):
        return f"{target_name.rsplit('.', 1)[0]}.weight"
    return target_name


def _convert_shape(source: np.ndarray, *, target_name: str, backend: BackendName) -> np.ndarray:
    if backend == "nnx" and target_name.endswith(".kernel"):
        if source.ndim == 2:
            return source.T
        if source.ndim == 4:
            return source.transpose(2, 3, 1, 0)
    if backend == "mlx" and target_name.endswith(".weight") and source.ndim == 4:
        return source.transpose(0, 2, 3, 1)
    return source


def convert_named_arrays(
    source: Mapping[str, np.ndarray],
    *,
    target_shapes: Mapping[str, tuple[int, ...]],
    backend: BackendName,
) -> dict[str, np.ndarray]:
    """Convert PyTorch arrays against a native model's named state schema."""
    expected_sources = {_source_name(name, backend=backend): name for name in target_shapes}
    missing = sorted(set(expected_sources) - set(source))
    unexpected = sorted(set(source) - set(expected_sources))
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing PyTorch arrays for targets: {missing}")
        if unexpected:
            details.append(f"unexpected PyTorch arrays: {unexpected}")
        raise CheckpointSchemaError(f"checkpoint schema mismatch for {backend}: {'; '.join(details)}")

    converted: dict[str, np.ndarray] = {}
    shape_errors: list[str] = []
    for source_name, target_name in expected_sources.items():
        value = _convert_shape(source[source_name], target_name=target_name, backend=backend)
        if tuple(value.shape) != tuple(target_shapes[target_name]):
            shape_errors.append(
                f"{target_name}: source {source_name} converted {tuple(value.shape)}, "
                f"expected {tuple(target_shapes[target_name])}"
            )
        converted[target_name] = value
    if shape_errors:
        raise CheckpointSchemaError(f"checkpoint shape mismatch for {backend}: {shape_errors}")
    return converted


def load_native_arrays(model: object, arrays: Mapping[str, np.ndarray], *, backend: BackendName) -> None:
    expected = native_state_arrays(model, backend=backend)
    expected_shapes = {name: tuple(value.shape) for name, value in expected.items()}
    actual_shapes = {name: tuple(value.shape) for name, value in arrays.items()}
    if set(actual_shapes) != set(expected_shapes):
        missing = sorted(set(expected_shapes) - set(actual_shapes))
        unexpected = sorted(set(actual_shapes) - set(expected_shapes))
        raise CheckpointSchemaError(
            f"native load schema mismatch for {backend}: missing={missing}, unexpected={unexpected}"
        )
    shape_errors = [
        f"{name}: got {shape}, expected {expected_shapes[name]}"
        for name, shape in actual_shapes.items()
        if shape != expected_shapes[name]
    ]
    if shape_errors:
        raise CheckpointSchemaError(f"native load shape mismatch for {backend}: {shape_errors}")

    if backend == "mlx":
        import mlx.core as mx
        import mlx.nn as nn

        assert isinstance(model, nn.Module)
        model.load_weights([(name, mx.array(value)) for name, value in arrays.items()], strict=True)
        return

    import jax.numpy as jnp
    from flax import nnx

    flat_state = nnx.to_flat_state(nnx.state(model))
    replacements = [(path, jnp.asarray(arrays[".".join(map(str, path))])) for path in flat_state.paths]
    nnx.update(model, nnx.from_flat_state(replacements))
