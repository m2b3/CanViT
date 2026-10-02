from typing import Final, cast

import jax
import jax.numpy as jnp

Initializer = jax.nn.initializers.Initializer
DINO_LINEAR_KERNEL: Final[Initializer] = jax.nn.initializers.truncated_normal(
    stddev=0.02,
    lower=-100.0,
    upper=100.0,
)


def symmetric_uniform(*, bound: float) -> Initializer:
    assert bound >= 0, bound

    def initialize(
        key: jax.Array,
        shape: tuple[int, ...],
        dtype: jax.typing.DTypeLike = jnp.float32,
        out_sharding: jax.sharding.NamedSharding | jax.P | None = None,
    ) -> jax.Array:
        return jax.random.uniform(
            key,
            shape,
            dtype=dtype,
            minval=-bound,
            maxval=bound,
            out_sharding=out_sharding,
        )

    return cast(Initializer, initialize)


def pytorch_linear_initializers(*, in_features: int) -> tuple[Initializer, Initializer]:
    bound = (1.0 / in_features) ** 0.5
    return symmetric_uniform(bound=bound), symmetric_uniform(bound=bound)
