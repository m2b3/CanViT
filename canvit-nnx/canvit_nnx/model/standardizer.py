import flax.nnx as nnx
import jax
import jax.numpy as jnp

Array = jax.Array


def _raise_unfitted(_: Array) -> None:
    raise RuntimeError("standardizer statistics were never fitted or loaded")


class PositionAwareStandardizer(nnx.Module):
    def __init__(self, *, n_positions: int, dim: int, eps: float = 1e-6) -> None:
        assert n_positions > 0 and dim > 0, (n_positions, dim)
        self.eps = eps
        self.mean = nnx.Variable(jnp.zeros((n_positions, dim), dtype=jnp.float32))
        self.var = nnx.Variable(jnp.ones((n_positions, dim), dtype=jnp.float32))
        self.fitted = nnx.Variable(jnp.asarray(False))

    def fit(self, samples: Array) -> None:
        assert samples.ndim == 3 and samples.shape[1:] == self.mean[...].shape, (
            samples.shape,
            self.mean[...].shape,
        )
        self.mean[...] = samples.mean(axis=0)
        self.var[...] = samples.var(axis=0)
        self.fitted[...] = jnp.asarray(True)

    @property
    def std(self) -> Array:
        return jnp.sqrt(self.var[...] + self.eps)

    def __call__(self, x: Array) -> Array:
        fitted = self.fitted[...]
        try:
            fitted_value = bool(fitted)
        except jax.errors.TracerBoolConversionError:

            def standardize(_: None) -> Array:
                return (x - self.mean[...]) / self.std

            def fail(_: None) -> Array:
                jax.debug.callback(_raise_unfitted, jnp.asarray(0, dtype=jnp.int32))
                return jnp.full_like(x, jnp.nan)

            return jax.lax.cond(fitted, standardize, fail, operand=None)
        if not fitted_value:
            raise RuntimeError("standardizer statistics were never fitted or loaded")
        return (x - self.mean[...]) / self.std

    def destandardize(self, x: Array) -> Array:
        return x * self.std + self.mean[...]
