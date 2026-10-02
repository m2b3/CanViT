from dataclasses import replace
from typing import Any

import canvit_pytorch
import numpy as np
import pytest
import torch
from canvit_core import CanViTConfig
from canvit_pytorch import Viewpoint as TorchViewpoint
from canvit_pytorch import sample_at_viewpoint

from tests.backends.support import Backend, canvit_values, normalized_l2, relative_l2, tokenwise_normalized_l2


def test_public_model_exports_match_reference(backend: Backend) -> None:
    assert set(backend.module.__all__) == set(canvit_pytorch.__all__)


def _assert_outputs(reference_values: dict[str, np.ndarray], native_values: dict[str, np.ndarray]) -> None:
    assert set(reference_values) == set(native_values)
    for name, reference in reference_values.items():
        actual = native_values[name]
        assert reference.shape == actual.shape, name
        assert relative_l2(reference, actual) < 2e-3, (name, relative_l2(reference, actual))
        assert normalized_l2(reference, actual) < 2e-3, (name, normalized_l2(reference, actual))
        if reference.ndim >= 2:
            assert tokenwise_normalized_l2(reference, actual) < 2e-3, name


def _step_inputs(backend: Backend, *, step: int, batch_size: int = 2) -> tuple[torch.Tensor, Any, Any, Any]:
    centers = np.asarray([
        [[0.0, 0.0], [-0.34, 0.22]],
        [[0.24, -0.17], [-0.48, -0.19]],
        [[-0.2, 0.38], [0.31, -0.41]],
    ], dtype=np.float32)[step][:batch_size]
    scales = np.asarray([
        [0.79, 0.46],
        [0.55, 0.39],
        [0.43, 0.63],
    ], dtype=np.float32)[step][:batch_size]
    generator = np.random.default_rng(2026)
    scene = generator.normal(size=(batch_size, 23, 17, 3)).astype(np.float32)
    torch_scene = torch.from_numpy(scene).permute(0, 3, 1, 2)
    torch_viewpoint = TorchViewpoint(torch.from_numpy(centers), torch.from_numpy(scales))
    torch_glimpse = sample_at_viewpoint(spatial=torch_scene, viewpoint=torch_viewpoint, glimpse_size_px=8)
    native_viewpoint = backend.viewpoint(centers, scales)
    native_glimpse = backend.module.sample_at_viewpoint(
        spatial=backend.array(scene), viewpoint=native_viewpoint, glimpse_size_px=8,
    )
    return torch_glimpse, torch_viewpoint, native_glimpse, native_viewpoint


def test_multistep_output_and_canvas_readouts_match_pytorch(
    reference_pair: tuple[Any, Any], backend: Backend,
):
    reference, native = reference_pair
    torch_glimpse, torch_viewpoint, native_glimpse, native_viewpoint = _step_inputs(backend, step=0)
    torch_state = reference.init_state(batch_size=2, canvas_grid_size=4)
    native_state = native.init_state(batch_size=2, canvas_grid_size=4)
    for step in range(3):
        if step:
            torch_glimpse, torch_viewpoint, native_glimpse, native_viewpoint = _step_inputs(backend, step=step)
        with torch.inference_mode():
            torch_output = reference(glimpse=torch_glimpse, state=torch_state, viewpoint=torch_viewpoint)
        native_output = native(glimpse=native_glimpse, state=native_state, viewpoint=native_viewpoint)
        torch_values = canvit_values(torch_output, backend)
        native_values = canvit_values(native_output, backend)
        _assert_outputs(torch_values, native_values)
        torch_state = torch_output.state
        native_state = native_output.state
        torch_canvas = torch_values["canvas"][:, reference.config.num_canvas_registers:]
        native_canvas = native_values["canvas"][:, native.config.num_canvas_registers:]
        assert tokenwise_normalized_l2(torch_canvas, native_canvas) < 2e-3


def test_model_properties_and_canvas_helpers(
    reference_pair: tuple[Any, Any], backend: Backend, tiny_config: CanViTConfig,
):
    reference, native = reference_pair
    assert native.config == tiny_config
    assert native.backbone_dim == reference.backbone_dim == 32
    assert native.canvas_dim == reference.canvas_dim == 16
    assert native.patch_size == reference.patch_size == 4
    state = native.init_state(batch_size=2, canvas_grid_size=4)
    canvas = backend.to_numpy(state.canvas)
    assert canvas.shape == (2, tiny_config.num_canvas_registers + 16, tiny_config.canvas_dim)
    patches = backend.to_numpy(native.canvas_patches(state.canvas))
    grid = backend.to_numpy(native.canvas_patch_grid(state.canvas))
    np.testing.assert_array_equal(patches, canvas[:, tiny_config.num_canvas_registers:])
    np.testing.assert_array_equal(grid, patches.reshape(2, 4, 4, tiny_config.canvas_dim))


@pytest.mark.parametrize(
    "changes",
    [
        {"canvas_projections": "qkvo"},
        {"enable_vpe": False},
        {"enable_reads": False},
        {"num_canvas_registers": 0, "num_backbone_registers": 0},
        {
            "canvas_projections": "qkvo",
            "enable_vpe": False,
            "enable_reads": False,
            "num_canvas_registers": 0,
            "num_backbone_registers": 0,
        },
    ],
)
def test_architecture_variants_run_and_match(
    backend: Backend, tiny_config: CanViTConfig, changes: dict[str, object],
):
    config = replace(tiny_config, **changes)
    reference, native = backend.reference_pair(config)
    torch_glimpse, torch_viewpoint, native_glimpse, native_viewpoint = _step_inputs(backend, step=1)
    torch_state = reference.init_state(batch_size=2, canvas_grid_size=4)
    native_state = native.init_state(batch_size=2, canvas_grid_size=4)
    with torch.inference_mode():
        torch_output = reference(glimpse=torch_glimpse, state=torch_state, viewpoint=torch_viewpoint)
    native_output = native(glimpse=native_glimpse, state=native_state, viewpoint=native_viewpoint)
    _assert_outputs(canvit_values(torch_output, backend), canvit_values(native_output, backend))
    if not config.enable_vpe:
        assert native_output.vpe is None
        assert torch_output.vpe is None


def test_eager_and_compiled_outputs_match(reference_pair: tuple[Any, Any], backend: Backend):
    _, native = reference_pair
    _, _, native_glimpse, native_viewpoint = _step_inputs(backend, step=0)
    state = native.init_state(batch_size=2, canvas_grid_size=4)
    eager = native(glimpse=native_glimpse, state=state, viewpoint=native_viewpoint)
    if backend.name == "mlx":
        import mlx.core as mx

        def mlx_call(glimpse, canvas, recurrent_cls, centers, scales):
            return native(
                glimpse=glimpse,
                state=type(state)(canvas=canvas, recurrent_cls=recurrent_cls),
                viewpoint=type(native_viewpoint)(centers=centers, scales=scales),
            )

        compiled = mx.compile(mlx_call)
        actual = compiled(native_glimpse, state.canvas, state.recurrent_cls,
                          native_viewpoint.centers, native_viewpoint.scales)
        mx.eval([actual.state.canvas, actual.state.recurrent_cls, actual.glimpse_patches])
    else:
        from flax import nnx

        @nnx.jit
        def nnx_call(model, glimpse, state, viewpoint):
            return model(glimpse=glimpse, state=state, viewpoint=viewpoint)

        actual = nnx_call(native, native_glimpse, state, native_viewpoint)
    _assert_outputs(canvit_values(eager, backend), canvit_values(actual, backend))
