"""Canvas Attention fused with the canvas's LayerNorm and RoPE, as Pallas kernels for TPU.

XLA lowers a Canvas Attention into several canvas-sized arrays in HBM: the
layer-normalized canvas, its rotated halves, casts and layout copies. Each
kernel here reads the canvas once and computes, for one batch element in VMEM,
its LayerNorm, RoPE and a single-pass softmax; the backward kernels recompute
these instead of storing attention probabilities. Dense attention suits these
sizes: a batch element's keys and values fit in VMEM.

The kernels compute what CanvasAttentionRead and CanvasAttentionWrite compute
with identity canvas-side maps ("asymmetric" projections), with the same
rounding points: queries, keys, values and probabilities in the compute dtype;
LayerNorm, RoPE, softmax and every accumulation in float32; the canvas in its
own dtype. Each cast is a cotangent boundary in the backward. Off TPU the
kernels run in Pallas interpret mode, which is how the tests run them.
"""

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec

from canvit_nnx.model.rope import RoPE, rotate_half

Array = jax.Array

ROW_MULTIPLE = 16
"""Glimpse-token axes are zero-padded to a multiple of the bfloat16 sublane tile."""
MASK_VALUE = -0.7 * float(np.finfo(np.float32).max)
VMEM_LIMIT_BYTES = 48 * 1024 * 1024
"""The scoped VMEM the backward kernels need at a 32 × 32 canvas, above the TPU default."""
COMPUTE_DTYPES = (jnp.dtype(jnp.float32), jnp.dtype(jnp.bfloat16))


def use_fused_kernels(compute_dtype: jnp.dtype) -> bool:
    """Whether Canvas Attention runs as these kernels: in bfloat16 on TPU, where they were validated."""
    return compute_dtype == jnp.bfloat16 and jax.default_backend() == "tpu"


def _head(index: int, head_dim: int) -> slice:
    return slice(index * head_dim, (index + 1) * head_dim)


def _rotate_half_transpose(x: Array) -> Array:
    """The adjoint of rotate_half: (first, second) -> (second, -first)."""
    half = x.shape[-1] // 2
    return jnp.concatenate((x[..., half:], -x[..., :half]), axis=-1)


def _matmul(a: Array, b: Array) -> Array:
    return lax.dot_general(a, b, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)


def _matmul_transposed_b(a: Array, b: Array) -> Array:
    return lax.dot_general(a, b, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)


def _matmul_transposed_a(a: Array, b: Array) -> Array:
    return lax.dot_general(a, b, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)


def _rounded(x: Array, dtype: jnp.dtype) -> Array:
    return x.astype(dtype).astype(jnp.float32)


def _softmax(scores: Array) -> Array:
    exponentials = jnp.exp(scores - jnp.max(scores, axis=1, keepdims=True))
    return exponentials / jnp.sum(exponentials, axis=1, keepdims=True)


def _layer_norm_statistics(canvas_ref, *, num_heads: int, head_dim: int, epsilon: float) -> tuple[Array, Array]:
    """[N, 1] mean and inverse standard deviation of each canvas token, read one head's columns at a time.

    A centered float32 copy of the whole canvas exceeded the scoped VMEM at a
    64 × 64 canvas. The variance is centered all the same: E[x²] - E[x]²
    cancels catastrophically on canvas tokens whose mean is hundreds of times
    their spread, which trained canvases have.
    """
    num_tokens = canvas_ref.shape[0]
    total = jnp.zeros((num_tokens, 1), jnp.float32)
    for index in range(num_heads):
        total += jnp.sum(canvas_ref[:, _head(index, head_dim)].astype(jnp.float32), axis=1, keepdims=True)
    mean = total / (num_heads * head_dim)
    squares = jnp.zeros((num_tokens, 1), jnp.float32)
    for index in range(num_heads):
        centered = canvas_ref[:, _head(index, head_dim)].astype(jnp.float32) - mean
        squares += jnp.sum(centered * centered, axis=1, keepdims=True)
    return mean, lax.rsqrt(squares / (num_heads * head_dim) + epsilon)


def _layer_norm_input_cotangent(
    standardized: Array, standardized_cotangent: Array, *, inverse_std: Array, mean_cotangent: Array,
    mean_cotangent_times_standardized: Array,
) -> Array:
    """One head's cotangent of the LayerNorm input, given row means of the cotangent over every feature."""
    return inverse_std * (standardized_cotangent - mean_cotangent - standardized * mean_cotangent_times_standardized)


# Canvas Attention Read: glimpse queries over keys rope(LN(canvas)) and values LN(canvas).


def _read_forward_kernel(canvas_ref, queries_ref, ln_scale_ref, ln_bias_ref, sin_ref, cos_ref, output_ref, *, epsilon):
    _, num_heads, head_dim = queries_ref.shape
    compute_dtype = queries_ref.dtype
    mean, inverse_std = _layer_norm_statistics(canvas_ref, num_heads=num_heads, head_dim=head_dim, epsilon=epsilon)
    sin, cos = sin_ref[...], cos_ref[...]
    for index in range(num_heads):
        columns = _head(index, head_dim)
        standardized = (canvas_ref[:, columns].astype(jnp.float32) - mean) * inverse_std
        normalized = standardized * ln_scale_ref[columns] + ln_bias_ref[columns]
        keys = (normalized * cos + rotate_half(normalized) * sin).astype(compute_dtype)
        probabilities = _softmax(_matmul_transposed_b(queries_ref[:, index, :], keys) * head_dim**-0.5)
        output_ref[:, index, :] = _matmul(probabilities.astype(compute_dtype), normalized.astype(compute_dtype)).astype(
            output_ref.dtype
        )


def _read_backward_kernel(
    canvas_ref, queries_ref, cotangent_ref, ln_scale_ref, ln_bias_ref, sin_ref, cos_ref,
    canvas_cotangent_ref, queries_cotangent_ref, ln_scale_cotangent_ref, ln_bias_cotangent_ref,
    standardized_cotangent_ref, *, epsilon,
):
    _, num_heads, head_dim = queries_ref.shape
    compute_dtype = queries_ref.dtype
    scale = head_dim**-0.5
    mean, inverse_std = _layer_norm_statistics(canvas_ref, num_heads=num_heads, head_dim=head_dim, epsilon=epsilon)
    sin, cos = sin_ref[...], cos_ref[...]
    row_sum = jnp.zeros_like(mean)
    row_sum_times_standardized = jnp.zeros_like(mean)
    for index in range(num_heads):
        columns = _head(index, head_dim)
        standardized = (canvas_ref[:, columns].astype(jnp.float32) - mean) * inverse_std
        normalized = standardized * ln_scale_ref[columns] + ln_bias_ref[columns]
        keys = (normalized * cos + rotate_half(normalized) * sin).astype(compute_dtype)
        queries = queries_ref[:, index, :]
        cotangent = cotangent_ref[:, index, :]
        probabilities = _softmax(_matmul_transposed_b(queries, keys) * scale)
        # Cotangents are rounded to the compute dtype wherever the forward rounds the primal.
        probabilities_cotangent = _rounded(_matmul_transposed_b(cotangent, normalized.astype(compute_dtype)), compute_dtype)
        values_cotangent = _rounded(_matmul_transposed_a(probabilities.astype(compute_dtype), cotangent), compute_dtype)
        scores_cotangent = (
            probabilities
            * (probabilities_cotangent - jnp.sum(probabilities_cotangent * probabilities, axis=1, keepdims=True))
            * scale
        )
        queries_cotangent_ref[:, index, :] = _matmul(scores_cotangent, keys).astype(queries_cotangent_ref.dtype)
        keys_cotangent = _rounded(_matmul_transposed_a(scores_cotangent, queries), compute_dtype)
        normalized_cotangent = values_cotangent + keys_cotangent * cos + _rotate_half_transpose(keys_cotangent * sin)
        ln_scale_cotangent_ref[0, columns] = jnp.sum(normalized_cotangent * standardized, axis=0)
        ln_bias_cotangent_ref[0, columns] = jnp.sum(normalized_cotangent, axis=0)
        standardized_cotangent = normalized_cotangent * ln_scale_ref[columns]
        row_sum += jnp.sum(standardized_cotangent, axis=1, keepdims=True)
        row_sum_times_standardized += jnp.sum(standardized_cotangent * standardized, axis=1, keepdims=True)
        standardized_cotangent_ref[:, columns] = standardized_cotangent
    # The LayerNorm pullback needs row means over every feature, so it runs once every head is staged.
    for index in range(num_heads):
        columns = _head(index, head_dim)
        standardized = (canvas_ref[:, columns].astype(jnp.float32) - mean) * inverse_std
        canvas_cotangent_ref[:, columns] = _layer_norm_input_cotangent(
            standardized, standardized_cotangent_ref[:, columns], inverse_std=inverse_std,
            mean_cotangent=row_sum / (num_heads * head_dim),
            mean_cotangent_times_standardized=row_sum_times_standardized / (num_heads * head_dim),
        ).astype(canvas_cotangent_ref.dtype)


# Canvas Attention Write: queries rope(LN(canvas)) over glimpse keys and values; the update is added to the canvas.


def _write_forward_kernel(
    canvas_ref, keys_ref, values_ref, ln_scale_ref, ln_bias_ref, sin_ref, cos_ref, output_ref, *, num_keys, epsilon
):
    num_padded_keys, num_heads, head_dim = keys_ref.shape
    compute_dtype = keys_ref.dtype
    mean, inverse_std = _layer_norm_statistics(canvas_ref, num_heads=num_heads, head_dim=head_dim, epsilon=epsilon)
    sin, cos = sin_ref[...], cos_ref[...]
    padding_mask = jnp.where(lax.broadcasted_iota(jnp.int32, (1, num_padded_keys), 1) < num_keys, 0.0, MASK_VALUE)
    for index in range(num_heads):
        columns = _head(index, head_dim)
        canvas_head = canvas_ref[:, columns].astype(jnp.float32)
        normalized = (canvas_head - mean) * inverse_std * ln_scale_ref[columns] + ln_bias_ref[columns]
        queries = (normalized * cos + rotate_half(normalized) * sin).astype(compute_dtype)
        scores = _matmul_transposed_b(queries, keys_ref[:, index, :]) * head_dim**-0.5 + padding_mask
        update = _matmul(_softmax(scores).astype(compute_dtype), values_ref[:, index, :])
        # Each head's columns are read before they are written, which makes the aliased output safe.
        output_ref[:, columns] = (canvas_head + update).astype(output_ref.dtype)


def _write_backward_kernel(
    canvas_ref, keys_ref, values_ref, cotangent_ref, ln_scale_ref, ln_bias_ref, sin_ref, cos_ref,
    canvas_cotangent_ref, keys_cotangent_ref, values_cotangent_ref, ln_scale_cotangent_ref, ln_bias_cotangent_ref,
    standardized_cotangent_ref, *, num_keys, epsilon,
):
    num_padded_keys, num_heads, head_dim = keys_ref.shape
    compute_dtype = keys_ref.dtype
    scale = head_dim**-0.5
    mean, inverse_std = _layer_norm_statistics(canvas_ref, num_heads=num_heads, head_dim=head_dim, epsilon=epsilon)
    sin, cos = sin_ref[...], cos_ref[...]
    valid_keys = lax.broadcasted_iota(jnp.int32, (1, num_padded_keys), 1) < num_keys
    padding_mask = jnp.where(valid_keys, 0.0, MASK_VALUE)
    row_sum = jnp.zeros_like(mean)
    row_sum_times_standardized = jnp.zeros_like(mean)
    # The update enters the residual sum in float32, so the cotangent reaches the attention products unrounded;
    # the rounding points are the queries and the probabilities.
    for index in range(num_heads):
        columns = _head(index, head_dim)
        standardized = (canvas_ref[:, columns].astype(jnp.float32) - mean) * inverse_std
        normalized = standardized * ln_scale_ref[columns] + ln_bias_ref[columns]
        queries = (normalized * cos + rotate_half(normalized) * sin).astype(compute_dtype)
        keys = keys_ref[:, index, :]
        values = values_ref[:, index, :]
        cotangent = cotangent_ref[:, columns]
        probabilities = jnp.where(
            valid_keys, _softmax(_matmul_transposed_b(queries, keys) * scale + padding_mask), 0.0
        )
        probabilities_cotangent = _rounded(_matmul_transposed_b(cotangent, values), compute_dtype)
        values_cotangent = _matmul_transposed_a(probabilities.astype(compute_dtype), cotangent).astype(compute_dtype)
        scores_cotangent = probabilities * (
            probabilities_cotangent - jnp.sum(probabilities_cotangent * probabilities, axis=1, keepdims=True)
        )
        scores_cotangent = jnp.where(valid_keys, scores_cotangent, 0.0) * scale
        queries_cotangent = _rounded(_matmul(scores_cotangent, keys), compute_dtype)
        keys_cotangent = _matmul_transposed_a(scores_cotangent, queries).astype(compute_dtype)
        # Padded key rows are outside the attention's input: their cotangents are exactly zero.
        valid_rows = jnp.broadcast_to(valid_keys.T, keys_cotangent.shape)
        keys_cotangent_ref[:, index, :] = jnp.where(valid_rows, keys_cotangent, jnp.zeros_like(keys_cotangent))
        values_cotangent_ref[:, index, :] = jnp.where(valid_rows, values_cotangent, jnp.zeros_like(values_cotangent))
        normalized_cotangent = queries_cotangent * cos + _rotate_half_transpose(queries_cotangent * sin)
        ln_scale_cotangent_ref[0, columns] = jnp.sum(normalized_cotangent * standardized, axis=0)
        ln_bias_cotangent_ref[0, columns] = jnp.sum(normalized_cotangent, axis=0)
        standardized_cotangent = normalized_cotangent * ln_scale_ref[columns]
        row_sum += jnp.sum(standardized_cotangent, axis=1, keepdims=True)
        row_sum_times_standardized += jnp.sum(standardized_cotangent * standardized, axis=1, keepdims=True)
        standardized_cotangent_ref[:, columns] = standardized_cotangent
    # The LayerNorm pullback needs row means over every feature, so it runs once every head is staged; the
    # residual sum adds the incoming cotangent, and the total is rounded once.
    for index in range(num_heads):
        columns = _head(index, head_dim)
        standardized = (canvas_ref[:, columns].astype(jnp.float32) - mean) * inverse_std
        layer_norm_cotangent = _layer_norm_input_cotangent(
            standardized, standardized_cotangent_ref[:, columns], inverse_std=inverse_std,
            mean_cotangent=row_sum / (num_heads * head_dim),
            mean_cotangent_times_standardized=row_sum_times_standardized / (num_heads * head_dim),
        )
        canvas_cotangent_ref[:, columns] = (cotangent_ref[:, columns].astype(jnp.float32) + layer_norm_cotangent).astype(
            canvas_cotangent_ref.dtype
        )


# pallas_call wiring: one grid step per batch element; parameter cotangents are summed over the batch outside.


def _pallas_call(
    kernel: Callable[..., None],
    *,
    batched: tuple[Array, ...],
    shared: tuple[Array, ...],
    out_shape: jax.ShapeDtypeStruct | tuple[jax.ShapeDtypeStruct, ...],
    name: str,
    interpret: bool,
    scratch_shapes: tuple[pl.MemoryRef, ...] = (),
    input_output_aliases: dict[int, int] | None = None,
) -> Array | tuple[Array, ...]:
    """Run kernel once per batch element on its slice of the batched arrays and every output; shared arrays are whole."""

    def per_example(shape: tuple[int, ...]) -> pl.BlockSpec:
        return pl.BlockSpec((None, *shape[1:]), lambda b: (b,) + (0,) * (len(shape) - 1))

    def whole(shape: tuple[int, ...]) -> pl.BlockSpec:
        return pl.BlockSpec(shape, lambda b: (0,) * len(shape))

    return pl.pallas_call(
        kernel,
        grid=(batched[0].shape[0],),
        in_specs=[*(per_example(x.shape) for x in batched), *(whole(x.shape) for x in shared)],
        out_specs=jax.tree.map(lambda output: per_example(output.shape), out_shape),
        out_shape=out_shape,
        scratch_shapes=scratch_shapes,
        input_output_aliases=input_output_aliases or {},
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",), vmem_limit_bytes=VMEM_LIMIT_BYTES),
        interpret=interpret,
        name=name,
    )(*batched, *shared)


def _parameter_cotangent_shape(batch_size: int, feature_dim: int) -> jax.ShapeDtypeStruct:
    # [B, 1, D] rather than [B, D]: a (1, D) block of a rank-2 array is neither sublane-aligned nor the full batch.
    return jax.ShapeDtypeStruct((batch_size, 1, feature_dim), jnp.float32)


def _read_forward(canvas, queries, ln_scale, ln_bias, sin, cos, *, epsilon, interpret):
    return _pallas_call(
        functools.partial(_read_forward_kernel, epsilon=epsilon),
        batched=(canvas, queries),
        shared=(ln_scale, ln_bias, sin, cos),
        out_shape=jax.ShapeDtypeStruct(queries.shape, queries.dtype),
        name="canvas_attention_read_forward",
        interpret=interpret,
    )


def _read_backward(canvas, queries, cotangent, ln_scale, ln_bias, sin, cos, *, epsilon, interpret):
    batch_size, num_tokens, feature_dim = canvas.shape
    canvas_cotangent, queries_cotangent, ln_scale_cotangents, ln_bias_cotangents = _pallas_call(
        functools.partial(_read_backward_kernel, epsilon=epsilon),
        batched=(canvas, queries, cotangent),
        shared=(ln_scale, ln_bias, sin, cos),
        out_shape=(
            jax.ShapeDtypeStruct(canvas.shape, canvas.dtype),
            jax.ShapeDtypeStruct(queries.shape, queries.dtype),
            _parameter_cotangent_shape(batch_size, feature_dim),
            _parameter_cotangent_shape(batch_size, feature_dim),
        ),
        scratch_shapes=(pltpu.VMEM((num_tokens, feature_dim), jnp.float32),),
        name="canvas_attention_read_backward",
        interpret=interpret,
    )
    return canvas_cotangent, queries_cotangent, ln_scale_cotangents.sum(axis=(0, 1)), ln_bias_cotangents.sum(axis=(0, 1))


@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7))
def _read(canvas, queries, ln_scale, ln_bias, sin, cos, epsilon, interpret):
    return _read_forward(canvas, queries, ln_scale, ln_bias, sin, cos, epsilon=epsilon, interpret=interpret)


def _read_forward_rule(canvas, queries, ln_scale, ln_bias, sin, cos, epsilon, interpret):
    output = _read_forward(canvas, queries, ln_scale, ln_bias, sin, cos, epsilon=epsilon, interpret=interpret)
    return output, (canvas, queries, ln_scale, ln_bias, sin, cos)


def _read_backward_rule(epsilon, interpret, residuals, cotangent):
    canvas, queries, ln_scale, ln_bias, sin, cos = residuals
    cotangents = _read_backward(
        canvas, queries, cotangent.astype(queries.dtype), ln_scale, ln_bias, sin, cos,
        epsilon=epsilon, interpret=interpret,
    )
    return (*cotangents, None, None)


_read.defvjp(_read_forward_rule, _read_backward_rule)


def _write_forward(canvas, keys, values, ln_scale, ln_bias, sin, cos, *, num_keys, epsilon, interpret):
    return _pallas_call(
        functools.partial(_write_forward_kernel, num_keys=num_keys, epsilon=epsilon),
        batched=(canvas, keys, values),
        shared=(ln_scale, ln_bias, sin, cos),
        out_shape=jax.ShapeDtypeStruct(canvas.shape, canvas.dtype),
        input_output_aliases={0: 0},
        name="canvas_attention_write_forward",
        interpret=interpret,
    )


def _write_backward(canvas, keys, values, cotangent, ln_scale, ln_bias, sin, cos, *, num_keys, epsilon, interpret):
    batch_size, num_tokens, feature_dim = canvas.shape
    canvas_cotangent, keys_cotangent, values_cotangent, ln_scale_cotangents, ln_bias_cotangents = _pallas_call(
        functools.partial(_write_backward_kernel, num_keys=num_keys, epsilon=epsilon),
        batched=(canvas, keys, values, cotangent),
        shared=(ln_scale, ln_bias, sin, cos),
        out_shape=(
            jax.ShapeDtypeStruct(canvas.shape, canvas.dtype),
            jax.ShapeDtypeStruct(keys.shape, keys.dtype),
            jax.ShapeDtypeStruct(values.shape, values.dtype),
            _parameter_cotangent_shape(batch_size, feature_dim),
            _parameter_cotangent_shape(batch_size, feature_dim),
        ),
        scratch_shapes=(pltpu.VMEM((num_tokens, feature_dim), jnp.float32),),
        name="canvas_attention_write_backward",
        interpret=interpret,
    )
    return (
        canvas_cotangent,
        keys_cotangent,
        values_cotangent,
        ln_scale_cotangents.sum(axis=(0, 1)),
        ln_bias_cotangents.sum(axis=(0, 1)),
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9))
def _write(canvas, keys, values, ln_scale, ln_bias, sin, cos, num_keys, epsilon, interpret):
    return _write_forward(
        canvas, keys, values, ln_scale, ln_bias, sin, cos, num_keys=num_keys, epsilon=epsilon, interpret=interpret
    )


def _write_forward_rule(canvas, keys, values, ln_scale, ln_bias, sin, cos, num_keys, epsilon, interpret):
    output = _write_forward(
        canvas, keys, values, ln_scale, ln_bias, sin, cos, num_keys=num_keys, epsilon=epsilon, interpret=interpret
    )
    return output, (canvas, keys, values, ln_scale, ln_bias, sin, cos)


def _write_backward_rule(num_keys, epsilon, interpret, residuals, cotangent):
    canvas, keys, values, ln_scale, ln_bias, sin, cos = residuals
    cotangents = _write_backward(
        canvas, keys, values, cotangent.astype(canvas.dtype), ln_scale, ln_bias, sin, cos,
        num_keys=num_keys, epsilon=epsilon, interpret=interpret,
    )
    return (*cotangents, None, None)


_write.defvjp(_write_forward_rule, _write_backward_rule)


# Public entry points: padding, RoPE tables, and the split of the batch over a data-parallel mesh.


def _canvas_rope_tables(rope: RoPE, *, num_canvas_tokens: int) -> tuple[Array, Array]:
    """[N, head_dim] sin and cos for every canvas token, registers first; registers rotate by zero."""
    assert rope.sin.ndim == 4 and rope.sin.shape[0] == 1 and rope.sin.shape[2] == 1, rope.sin.shape
    num_registers = num_canvas_tokens - rope.sin.shape[1]
    assert num_registers >= 0, (num_canvas_tokens, rope.sin.shape)
    head_dim = rope.sin.shape[-1]
    sin = jnp.concatenate((jnp.zeros((num_registers, head_dim), jnp.float32), rope.sin[0, :, 0, :]))
    cos = jnp.concatenate((jnp.ones((num_registers, head_dim), jnp.float32), rope.cos[0, :, 0, :]))
    return lax.stop_gradient(sin), lax.stop_gradient(cos)


def _pad_rows(x: Array) -> Array:
    """Zero-pad axis 1 of [B, N, H, d] to a multiple of ROW_MULTIPLE, outside the custom VJPs."""
    return lax.pad(x, x.dtype.type(0), ((0, 0, 0), (0, -x.shape[1] % ROW_MULTIPLE, 0), (0, 0, 0), (0, 0, 0)))


def _over_batch_shards(function: Callable[..., Array], arguments: tuple[Array, ...], batched: tuple[bool, ...]) -> Array:
    """Calls function on each device's share of the batch under a mesh: Pallas kernels are invisible to GSPMD."""
    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty:
        return function(*arguments)
    batch = PartitionSpec(mesh.axis_names)
    in_specs = tuple(batch if is_batched else PartitionSpec() for is_batched in batched)
    return jax.shard_map(function, in_specs=in_specs, out_specs=batch, check_vma=False)(*arguments)


def _check_operands(canvas: Array, glimpse_side: Array, ln_scale: Array, ln_bias: Array) -> None:
    batch_size, _, feature_dim = canvas.shape
    assert glimpse_side.ndim == 4 and glimpse_side.shape[0] == batch_size, (canvas.shape, glimpse_side.shape)
    _, _, num_heads, head_dim = glimpse_side.shape
    assert num_heads * head_dim == feature_dim and head_dim % 2 == 0, (canvas.shape, glimpse_side.shape)
    assert canvas.dtype in COMPUTE_DTYPES and glimpse_side.dtype in COMPUTE_DTYPES, (canvas.dtype, glimpse_side.dtype)
    assert ln_scale.shape == ln_bias.shape == (feature_dim,), (ln_scale.shape, ln_bias.shape)
    assert ln_scale.dtype == ln_bias.dtype == jnp.float32, (ln_scale.dtype, ln_bias.dtype)


def fused_canvas_read(
    *, canvas: Array, queries: Array, ln_scale: Array, ln_bias: Array, epsilon: float, canvas_rope: RoPE
) -> Array:
    """[B, Nq, H, d] rotated queries attend over keys rope(LN(canvas)) and values LN(canvas) of a [B, N, H·d] canvas.

    Returns [B, Nq, H, d] in the queries' dtype. LN is the LayerNorm with
    ln_scale, ln_bias and epsilon; canvas_rope covers the canvas patches,
    which follow the registers.
    """
    _check_operands(canvas, queries, ln_scale, ln_bias)
    sin, cos = _canvas_rope_tables(canvas_rope, num_canvas_tokens=canvas.shape[1])
    interpret = jax.default_backend() != "tpu"

    def read(canvas: Array, queries: Array, ln_scale: Array, ln_bias: Array, sin: Array, cos: Array) -> Array:
        return _read(canvas, queries, ln_scale, ln_bias, sin, cos, epsilon, interpret)

    output = _over_batch_shards(
        read, (canvas, _pad_rows(queries), ln_scale, ln_bias, sin, cos), batched=(True, True, False, False, False, False)
    )
    return output[:, : queries.shape[1]]


def fused_canvas_write(
    *, canvas: Array, keys: Array, values: Array, ln_scale: Array, ln_bias: Array, epsilon: float, canvas_rope: RoPE
) -> Array:
    """Queries rope(LN(canvas)) attend over [B, Nk, H, d] keys (rotated) and values; returns canvas + attention output.

    The sum is computed in float32 and rounded once to the canvas dtype.
    """
    _check_operands(canvas, keys, ln_scale, ln_bias)
    assert values.shape == keys.shape and values.dtype == keys.dtype, (keys.shape, values.shape, keys.dtype, values.dtype)
    sin, cos = _canvas_rope_tables(canvas_rope, num_canvas_tokens=canvas.shape[1])
    interpret = jax.default_backend() != "tpu"
    num_keys = keys.shape[1]

    def write(canvas: Array, keys: Array, values: Array, ln_scale: Array, ln_bias: Array, sin: Array, cos: Array) -> Array:
        return _write(canvas, keys, values, ln_scale, ln_bias, sin, cos, num_keys, epsilon, interpret)

    return _over_batch_shards(
        write,
        (canvas, _pad_rows(keys), _pad_rows(values), ln_scale, ln_bias, sin, cos),
        batched=(True, True, True, False, False, False, False),
    )


__all__ = ["fused_canvas_read", "fused_canvas_write", "use_fused_kernels"]
