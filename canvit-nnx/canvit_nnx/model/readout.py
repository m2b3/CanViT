import flax.nnx as nnx
import jax

from canvit_nnx.model.initialization import pytorch_linear_initializers

Array = jax.Array


class LinearReadout(nnx.Module):
    def __init__(self, *, in_dim: int, out_dim: int, rngs: nnx.Rngs) -> None:
        self.norm = nnx.LayerNorm(in_dim, epsilon=1e-5, use_fast_variance=False, rngs=rngs)
        kernel_init, bias_init = pytorch_linear_initializers(in_features=in_dim)
        self.proj = nnx.Linear(
            in_features=in_dim,
            out_features=out_dim,
            kernel_init=kernel_init,
            bias_init=bias_init,
            rngs=rngs,
        )

    def __call__(self, x: Array) -> Array:
        return self.proj(self.norm(x))
