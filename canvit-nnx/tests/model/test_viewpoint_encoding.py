import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from canvit_nnx import Viewpoint
from canvit_nnx.model.vpe import ViewpointEncoding


def test_phases_are_a_float32_product_whatever_the_default_matmul_precision() -> None:
    """A CPU computes float32 products exactly at any precision setting, so a numeric comparison here could not
    catch the defect this guards against; the traced product's declared precision can."""
    encoding = ViewpointEncoding(dim=64, rngs=nnx.Rngs(0))
    viewpoint = Viewpoint(centers=jnp.array([[0.17, -0.23]], jnp.float32), scales=jnp.array([0.05], jnp.float32))
    with jax.default_matmul_precision("bfloat16"):
        products = [
            equation
            for equation in jax.make_jaxpr(encoding)(viewpoint).eqns
            if equation.primitive.name == "dot_general"
        ]
    assert len(products) == 1, products
    (product,) = products
    assert [getattr(operand.aval, "dtype", None) for operand in product.invars] == [jnp.float32, jnp.float32]
    assert product.params["precision"] == (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)


def test_encoding_matches_a_float64_reference() -> None:
    encoding = ViewpointEncoding(dim=64, rngs=nnx.Rngs(0))
    centers = np.array([[0.17, -0.23], [-0.9, 0.4]], np.float32)
    scales = np.array([0.05, 1.0], np.float32)
    actual = encoding(Viewpoint(centers=jnp.asarray(centers), scales=jnp.asarray(scales)))

    coordinates = np.concatenate((centers / scales[:, None], np.log(scales)[:, None]), axis=-1).astype(np.float64)
    phases = coordinates @ np.asarray(encoding.frequencies[...], np.float64).T
    features = np.concatenate((np.cos(phases), np.sin(phases)), axis=-1)
    normalized = (features - features.mean(-1, keepdims=True)) / np.sqrt(features.var(-1, keepdims=True) + 1e-5)
    np.testing.assert_allclose(actual, normalized / np.sqrt(64), atol=2e-6)
