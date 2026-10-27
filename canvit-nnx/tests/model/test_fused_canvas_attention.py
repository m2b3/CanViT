"""The fused Canvas Attention kernels, in Pallas interpret mode, against the pure-JAX Canvas Attention.

Interpret mode checks numerics only: whether the kernels compile and fit in
VMEM can only be checked on a TPU.
"""

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec
from support import relative_l2

from canvit_nnx.model import fused_canvas_attention
from canvit_nnx.model.attention import CanvasAttentionRead, CanvasAttentionWrite
from canvit_nnx.model.config import ComputeDtype
from canvit_nnx.model.rope import RoPE, compute_2d_rope, make_rope_periods
from canvit_nnx.viewpoint import grid_coords

# A CanViT-B glimpse at an 8 × 8 canvas: 8 heads of 128, 16 canvas registers, 7 glimpse tokens before 64 patches.
BACKBONE_DIM, NUM_HEADS, HEAD_DIM = 768, 8, 128
CANVAS_GRID_SIZE, NUM_CANVAS_REGISTERS = 8, 16
NUM_PREFIX_TOKENS, GLIMPSE_GRID_SIZE = 7, 8
BATCH_SIZE = 2

Kind = Literal["read", "write"]
Inputs = tuple[jax.Array, jax.Array, RoPE, RoPE]


def _module(kind: Kind, compute_dtype: ComputeDtype) -> CanvasAttentionRead | CanvasAttentionWrite:
    module_class = CanvasAttentionRead if kind == "read" else CanvasAttentionWrite
    module = module_class(
        backbone_dim=BACKBONE_DIM,
        canvas_dim=NUM_HEADS * HEAD_DIM,
        num_heads=NUM_HEADS,
        projections="asymmetric",
        dtype=jnp.dtype(compute_dtype),
        rngs=nnx.Rngs(0),
    )
    # LayerNorms away from their initial identity, so that their parameters matter.
    for index, norm in enumerate((module.ln_q, module.ln_kv)):
        assert norm.scale is not None and norm.bias is not None
        keys = jax.random.split(jax.random.key(index), 2)
        norm.scale[...] = 1.0 + 0.2 * jax.random.normal(keys[0], norm.scale[...].shape)
        norm.bias[...] = 0.1 * jax.random.normal(keys[1], norm.bias[...].shape)
    return module


def _inputs(compute_dtype: ComputeDtype) -> Inputs:
    rng = np.random.default_rng(0)
    canvas_tokens = NUM_CANVAS_REGISTERS + CANVAS_GRID_SIZE**2
    # An offset canvas, as trained canvases are: LayerNorm statistics must not cancel.
    canvas = rng.normal(loc=0.5, scale=2.0, size=(BATCH_SIZE, canvas_tokens, NUM_HEADS * HEAD_DIM))
    tokens = rng.normal(size=(BATCH_SIZE, NUM_PREFIX_TOKENS + GLIMPSE_GRID_SIZE**2, BACKBONE_DIM))
    periods = make_rope_periods(head_dim=HEAD_DIM, base=100.0)
    canvas_rope = compute_2d_rope(positions=grid_coords(size=CANVAS_GRID_SIZE).reshape(1, -1, 2), periods=periods)
    centers = rng.uniform(-0.3, 0.3, size=(BATCH_SIZE, 1, 1, 2))
    glimpse_positions = centers + 0.6 * np.asarray(grid_coords(size=GLIMPSE_GRID_SIZE))[None]
    tokens_rope = compute_2d_rope(
        positions=jnp.asarray(glimpse_positions.reshape(BATCH_SIZE, -1, 2), dtype=jnp.float32), periods=periods
    )
    dtype = jnp.dtype(compute_dtype)
    return jnp.asarray(canvas, dtype=dtype), jnp.asarray(tokens, dtype=dtype), canvas_rope, tokens_rope


def _apply(module: CanvasAttentionRead | CanvasAttentionWrite, canvas: jax.Array, tokens: jax.Array,
           canvas_rope: RoPE, tokens_rope: RoPE) -> jax.Array:
    if isinstance(module, CanvasAttentionRead):
        return module(tokens=tokens, canvas=canvas, tokens_rope=tokens_rope, canvas_rope=canvas_rope)
    return module(canvas=canvas, tokens=tokens, canvas_rope=canvas_rope, tokens_rope=tokens_rope)


def _output_and_gradients(module: CanvasAttentionRead | CanvasAttentionWrite, inputs: Inputs) -> dict[str, jax.Array]:
    """The output, and the gradients of its inner product with a fixed cotangent, by input and parameter."""
    canvas, tokens, canvas_rope, tokens_rope = inputs
    output = _apply(module, canvas, tokens, canvas_rope, tokens_rope)
    cotangent = jax.random.normal(jax.random.key(7), output.shape)

    def objective(module, canvas, tokens):
        return jnp.sum(_apply(module, canvas, tokens, canvas_rope, tokens_rope).astype(jnp.float32) * cotangent)

    parameters, canvas_gradient, tokens_gradient = nnx.grad(
        objective, argnums=(nnx.DiffState(0, nnx.Param), 1, 2)
    )(module, canvas, tokens)
    flat = nnx.to_flat_state(parameters)
    return {"output": output, "canvas": canvas_gradient, "tokens": tokens_gradient} | {
        ".".join(map(str, path)): leaf[...] for path, leaf in zip(flat.paths, flat.leaves, strict=True)
    }


def _evaluate(kind: Kind, compute_dtype: ComputeDtype, *, fused: bool, monkeypatch: pytest.MonkeyPatch) -> dict[str, jax.Array]:
    with monkeypatch.context() as patch:
        patch.setattr(fused_canvas_attention, "use_fused_kernels", lambda _: fused)
        return _output_and_gradients(_module(kind, compute_dtype), _inputs(compute_dtype))


@pytest.mark.parametrize("kind", ["read", "write"])
def test_fused_kernels_compute_canvas_attention_and_its_gradients(kind: Kind, monkeypatch: pytest.MonkeyPatch) -> None:
    reference = _evaluate(kind, "float32", fused=False, monkeypatch=monkeypatch)
    fused = _evaluate(kind, "float32", fused=True, monkeypatch=monkeypatch)
    assert fused.keys() == reference.keys()
    for name, expected in reference.items():
        assert fused[name].dtype == expected.dtype, name
        # Measured: at most 5.5e-7, from the order of float32 accumulation.
        assert relative_l2(expected, fused[name]) < 1e-5, (name, relative_l2(expected, fused[name]))


@pytest.mark.parametrize("kind", ["read", "write"])
def test_fused_kernels_round_like_canvas_attention_in_bfloat16(kind: Kind, monkeypatch: pytest.MonkeyPatch) -> None:
    exact = _evaluate(kind, "float32", fused=False, monkeypatch=monkeypatch)
    reference = _evaluate(kind, "bfloat16", fused=False, monkeypatch=monkeypatch)
    fused = _evaluate(kind, "bfloat16", fused=True, monkeypatch=monkeypatch)
    for name, expected in reference.items():
        assert fused[name].dtype == expected.dtype, name
        # Rounding in different places would move the kernels farther from float32 than Canvas Attention is.
        # Measured: the ratio of the two distances is between 0.95 and 1.02; the distances are at most 1.7e-2.
        fused_error = relative_l2(exact[name], fused[name])
        reference_error = relative_l2(exact[name], expected)
        assert fused_error <= 1.25 * reference_error + 1e-6, (name, fused_error, reference_error)
        assert fused_error < 5e-2, (name, fused_error)


@pytest.mark.parametrize("kind", ["read", "write"])
def test_fused_kernels_split_the_batch_over_a_mesh(kind: Kind, monkeypatch: pytest.MonkeyPatch) -> None:
    assert jax.device_count() >= BATCH_SIZE, "tests/conftest.py sets XLA_FLAGS for two CPU devices"
    single_device = _evaluate(kind, "float32", fused=True, monkeypatch=monkeypatch)
    mesh = jax.make_mesh((BATCH_SIZE,), ("batch",), axis_types=(jax.sharding.AxisType.Auto,))
    canvas, tokens, canvas_rope, tokens_rope = _inputs("float32")
    with monkeypatch.context() as patch, jax.set_mesh(mesh):
        patch.setattr(fused_canvas_attention, "use_fused_kernels", lambda _: True)
        batch = NamedSharding(mesh, PartitionSpec("batch"))
        inputs = (jax.device_put(canvas, batch), jax.device_put(tokens, batch), canvas_rope, tokens_rope)
        sharded = nnx.jit(_output_and_gradients)(_module(kind, "float32"), inputs)
    for name, expected in single_device.items():
        # Parameter gradients sum over the batch: a cross-device sum missing or doubled would be off by 50 to 100%.
        # Measured: at most 6e-7.
        assert relative_l2(expected, sharded[name]) < 1e-5, (name, relative_l2(expected, sharded[name]))
