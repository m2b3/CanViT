"""ImageNet-1k fine-tuning steps in NNX against the same steps written with canvit-pytorch and torch.optim.

The PyTorch side follows canvit_pytorch.specialize.in1k_tpu.train.train_step with
full backpropagation through time; that module needs torch_xla to import.
"""

import dataclasses

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from canvit_core import CanViTConfig
from canvit_pytorch import CanViTForImageClassification as TorchClassifier
from canvit_pytorch import Viewpoint as TorchViewpoint
from canvit_pytorch import sample_at_viewpoint as torch_sample_at_viewpoint

from tests.backends.support import Backend, relative_l2
from tools.checkpoint_conversion import convert_named_arrays, native_state_arrays

BATCH_SIZE, NUM_CLASSES, CANVAS_GRID_SIZE = 2, 7, 4


def test_finetuning_steps_match_pytorch(backend: Backend, tiny_config: CanViTConfig) -> None:
    if backend.name != "nnx":
        pytest.skip("fine-tuning is implemented in canvit_nnx")
    import jax.numpy as jnp
    from flax import nnx

    from canvit_nnx.specialize.in1k.train import Recipe, optimizer_transform, train_step

    # Warmup ends after a step at learning rate 0; a large peak makes the second update measurable.
    recipe = dataclasses.replace(Recipe(), batch_size=BATCH_SIZE, warmup_steps=1, peak_lr=1e-2)
    torch.manual_seed(0)
    reference = TorchClassifier(canvit_config=tiny_config, n_classes=NUM_CLASSES)
    native = backend.make_classifier(tiny_config, NUM_CLASSES)
    backend.load_reference_weights(native, reference)

    rng = np.random.default_rng(0)
    scenes = rng.normal(size=(BATCH_SIZE, 40, 40, 3)).astype(np.float32)
    labels = np.asarray([3, 5])
    centers = np.asarray([[[0.0, 0.0], [0.0, 0.0]], [[-0.3, 0.2], [0.4, -0.1]], [[0.1, 0.5], [-0.5, -0.4]]], np.float32)
    scales = np.asarray([[1.0, 1.0], [0.5, 0.35], [0.6, 0.25]], np.float32)

    torch_optimizer = torch.optim.AdamW(reference.parameters(), lr=recipe.peak_lr, weight_decay=recipe.weight_decay)
    torch_scenes = torch.from_numpy(scenes).permute(0, 3, 1, 2)
    torch_results = []
    for learning_rate in (0.0, recipe.peak_lr):
        for group in torch_optimizer.param_groups:
            group["lr"] = learning_rate
        state = reference.init_state(batch_size=BATCH_SIZE, canvas_grid_size=CANVAS_GRID_SIZE)
        total = torch.zeros(())
        for glimpse_centers, glimpse_scales in zip(centers, scales, strict=True):
            viewpoint = TorchViewpoint(centers=torch.from_numpy(glimpse_centers), scales=torch.from_numpy(glimpse_scales))
            glimpse = torch_sample_at_viewpoint(spatial=torch_scenes, viewpoint=viewpoint, glimpse_size_px=128)
            logits, state = reference(glimpse=glimpse, state=state, viewpoint=viewpoint)
            total = total + F.cross_entropy(logits, torch.from_numpy(labels), label_smoothing=recipe.label_smoothing)
        loss = total / len(centers)
        torch_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.cat([p.grad.flatten() for p in reference.parameters() if p.grad is not None]).norm()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), recipe.max_grad_norm)
        torch_optimizer.step()
        torch_results.append((loss.item(), grad_norm.item()))

    native_optimizer = nnx.Optimizer(native, optimizer_transform(recipe), wrt=nnx.Param)
    viewpoints = tuple(backend.viewpoint(c, s) for c, s in zip(centers, scales, strict=True))
    for torch_loss, torch_grad_norm in torch_results:
        results = train_step(
            native, native_optimizer, jnp.asarray(scenes), jnp.asarray(labels), viewpoints,
            label_smoothing=recipe.label_smoothing, canvas_grid_size=CANVAS_GRID_SIZE,
        )
        assert abs(float(results.loss) - torch_loss) < 1e-5 * abs(torch_loss), (float(results.loss), torch_loss)
        assert abs(float(results.grad_norm) - torch_grad_norm) < 1e-4 * torch_grad_norm
        assert torch_grad_norm > recipe.max_grad_norm, "the clipping threshold is exercised"

    expected = convert_named_arrays(
        {name: parameter.detach().numpy() for name, parameter in reference.named_parameters()},
        target_shapes={name: value.shape for name, value in native_state_arrays(native, backend="nnx", trainable_only=True).items()},
        backend="nnx",
    )
    actual = native_state_arrays(native, backend="nnx", trainable_only=True)
    for name, value in expected.items():
        # Measured: at most 1.8e-5, median 1.5e-6. Adam divides each update by its gradient's magnitude, which
        # passes float32 noise in small gradients to the parameters.
        assert relative_l2(value, actual[name]) < 1e-4, (name, relative_l2(value, actual[name]))
