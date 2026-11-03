import jax
import jax.numpy as jnp
import numpy as np

from canvit_nnx.model.multihead import attention

BATCH_SIZE, NUM_HEADS, NUM_QUERIES, NUM_KEYS, HEAD_DIM = 2, 2, 24, 7, 16


def test_the_score_cotangent_is_computed_apart_from_the_products_that_consume_it() -> None:
    """XLA on TPU computed the score cotangent wrongly when it fused it into the products that consume it.

    That happens in TPU code generation, so no CPU computation reproduces it (the network test
    tests/specialize/test_in1k_gradients.py does, on TPU); this checks, on any backend, that the score cotangent
    reaches those products through an optimization barrier.
    """
    rng = np.random.default_rng(0)
    queries, keys, values = (
        jnp.asarray(rng.normal(size=(BATCH_SIZE, tokens, NUM_HEADS, HEAD_DIM)), jnp.bfloat16)
        for tokens in (NUM_QUERIES, NUM_KEYS, NUM_KEYS)
    )

    def objective(queries, keys):
        return jnp.sum(attention(queries, keys, values, probability_dtype=jnp.bfloat16))

    jaxpr = jax.make_jaxpr(jax.grad(objective, argnums=(0, 1)))(queries, keys).jaxpr
    score_shape = (BATCH_SIZE, NUM_HEADS, NUM_QUERIES, NUM_KEYS)
    barriers = [eqn for eqn in jaxpr.eqns if eqn.primitive.name == "optimization_barrier"]
    assert any(getattr(var.aval, "shape", None) == score_shape for eqn in barriers for var in eqn.outvars), barriers
