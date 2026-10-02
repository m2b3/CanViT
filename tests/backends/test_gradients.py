from typing import Any

import torch

from tests.backends.support import Backend, relative_l2
from tools.checkpoint_conversion import convert_named_arrays, native_gradient_arrays, native_state_arrays


def _loss_torch(model: Any, glimpse: Any, viewpoint: Any, target: Any):
    state = model.init_state(batch_size=glimpse.shape[0], canvas_grid_size=4)
    for _ in range(2):
        output = model(glimpse=glimpse, state=state, viewpoint=viewpoint)
        state = output.state
    canvas = state.canvas
    canvas_mean = canvas.mean(dim=-1, keepdim=True)
    canvas_std = canvas.var(dim=-1, keepdim=True, unbiased=False).add(1e-5).sqrt()
    recurrent_cls = state.recurrent_cls
    cls_mean = recurrent_cls.mean(dim=-1, keepdim=True)
    cls_std = recurrent_cls.var(dim=-1, keepdim=True, unbiased=False).add(1e-5).sqrt()
    values = torch.cat([
        ((canvas - canvas_mean) / canvas_std).flatten(),
        ((recurrent_cls - cls_mean) / cls_std).flatten(),
    ])
    return ((values - target) ** 2).mean()


def _loss_native(model: Any, glimpse: Any, viewpoint: Any, target: Any, backend: Backend):
    state = model.init_state(batch_size=glimpse.shape[0], canvas_grid_size=4)
    for _ in range(2):
        output = model(glimpse=glimpse, state=state, viewpoint=viewpoint)
        state = output.state
    if backend.name == "mlx":
        import mlx.core as mx

        canvas = state.canvas
        canvas_mean = mx.mean(canvas, axis=-1, keepdims=True)
        canvas_std = mx.sqrt(mx.var(canvas, axis=-1, keepdims=True) + 1e-5)
        recurrent_cls = state.recurrent_cls
        cls_mean = mx.mean(recurrent_cls, axis=-1, keepdims=True)
        cls_std = mx.sqrt(mx.var(recurrent_cls, axis=-1, keepdims=True) + 1e-5)
        values = mx.concatenate([
            ((canvas - canvas_mean) / canvas_std).reshape(-1),
            ((recurrent_cls - cls_mean) / cls_std).reshape(-1),
        ])
        return mx.mean((values - target) ** 2)
    import jax.numpy as jnp

    canvas = state.canvas
    canvas_mean = jnp.mean(canvas, axis=-1, keepdims=True)
    canvas_std = jnp.sqrt(jnp.var(canvas, axis=-1, keepdims=True) + 1e-5)
    recurrent_cls = state.recurrent_cls
    cls_mean = jnp.mean(recurrent_cls, axis=-1, keepdims=True)
    cls_std = jnp.sqrt(jnp.var(recurrent_cls, axis=-1, keepdims=True) + 1e-5)
    values = jnp.concatenate([
        ((canvas - canvas_mean) / canvas_std).reshape(-1),
        ((recurrent_cls - cls_mean) / cls_std).reshape(-1),
    ])
    return jnp.mean((values - target) ** 2)


def test_input_and_parameter_gradients_match_pytorch(
    reference_pair: tuple[Any, Any], backend: Backend,
):
    reference, native = reference_pair
    torch_glimpse, torch_viewpoint, native_glimpse, native_viewpoint = backend.inputs()
    generator = torch.Generator().manual_seed(2718)
    initial_state = reference.init_state(batch_size=2, canvas_grid_size=4)
    target_torch = torch.randn((initial_state.canvas.numel() + initial_state.recurrent_cls.numel(),), generator=generator)
    target_native = backend.array(target_torch.numpy())
    torch_glimpse.requires_grad_(True)
    torch_loss = _loss_torch(reference, torch_glimpse, torch_viewpoint, target_torch)
    torch_loss.backward()
    torch_parameter_gradients = {
        name: parameter.grad.detach().numpy()
        for name, parameter in reference.named_parameters()
        if parameter.grad is not None
    }

    if backend.name == "mlx":
        import mlx.core as mx
        import mlx.nn as nn

        loss_and_grad = nn.value_and_grad(
            native,
            lambda glimpse: _loss_native(native, glimpse, native_viewpoint, target_native, backend),
        )
        _, native_parameter_gradients = loss_and_grad(native_glimpse)
        input_gradient = mx.grad(
            lambda glimpse: _loss_native(native, glimpse, native_viewpoint, target_native, backend)
        )(native_glimpse)
        native_parameter_gradients = native_gradient_arrays(native_parameter_gradients, backend=backend.name)
    else:
        from flax import nnx

        def loss(model, glimpse):
            return _loss_native(model, glimpse, native_viewpoint, target_native, backend)

        gradients, input_gradient = nnx.grad(
            loss,
            argnums=(nnx.DiffState(0, nnx.Param), 1),
        )(native, native_glimpse)
        native_parameter_gradients = native_gradient_arrays(gradients, backend=backend.name)

    trainable_schema = {
        name: tuple(value.shape)
        for name, value in native_state_arrays(native, backend=backend.name, trainable_only=True).items()
    }
    converted_gradients = convert_named_arrays(
        torch_parameter_gradients,
        target_shapes=trainable_schema,
        backend=backend.name,
    )
    assert "vpe.frequencies" not in trainable_schema
    assert "vpe.frequencies" not in native_parameter_gradients
    assert set(native_parameter_gradients) == set(converted_gradients)
    for name, expected in converted_gradients.items():
        difference = relative_l2(expected, native_parameter_gradients[name])
        assert difference < 5e-3, (name, difference)

    assert torch_glimpse.grad is not None
    torch_input_gradient = torch_glimpse.grad.detach().numpy().transpose(0, 2, 3, 1)
    assert relative_l2(torch_input_gradient, backend.to_numpy(input_gradient)) < 5e-3

    if backend.name == "mlx":
        import mlx.core as mx
        import mlx.nn as nn

        eager_loss_and_grad = nn.value_and_grad(
            native,
            lambda glimpse: _loss_native(native, glimpse, native_viewpoint, target_native, backend),
        )
        compiled_loss_and_grad = mx.compile(eager_loss_and_grad)
        eager_loss, eager_grad = eager_loss_and_grad(native_glimpse)
        compiled_loss, compiled_grad = compiled_loss_and_grad(native_glimpse)
        mx.eval([eager_loss, compiled_loss])
        assert abs(float(eager_loss) - float(compiled_loss)) < 1e-5
        eager_flat = native_gradient_arrays(eager_grad, backend=backend.name)
        compiled_flat = native_gradient_arrays(compiled_grad, backend=backend.name)
    else:
        from flax import nnx

        def loss(model, glimpse):
            return _loss_native(model, glimpse, native_viewpoint, target_native, backend)

        eager_grad_fn = nnx.grad(loss, argnums=(nnx.DiffState(0, nnx.Param), 1))
        compiled_grad_fn = nnx.jit(eager_grad_fn)
        eager_grad, eager_input = eager_grad_fn(native, native_glimpse)
        compiled_grad, compiled_input = compiled_grad_fn(native, native_glimpse)
        eager_flat = native_gradient_arrays(eager_grad, backend=backend.name)
        compiled_flat = native_gradient_arrays(compiled_grad, backend=backend.name)
        eager_input = backend.to_numpy(eager_input)
        compiled_input = backend.to_numpy(compiled_input)
        assert relative_l2(eager_input, compiled_input) < 5e-3
    for name in eager_flat:
        assert relative_l2(eager_flat[name], compiled_flat[name]) < 5e-3, name
