import jax
import jax.numpy as jnp
import numpy as np
from canvit_core import CanViTConfig
from flax import nnx
from support import relative_l2

from canvit_nnx import CanViTForImageClassification, Viewpoint, sample_at_viewpoint
from canvit_nnx.model.config import ComputeDtype

BATCH_SIZE = 3
CANVAS_GRID_SIZE = 4


def _episode(seed: int) -> tuple[jax.Array, list[Viewpoint], jax.Array]:
    rng = np.random.default_rng(seed)
    scenes = jnp.asarray(rng.normal(size=(BATCH_SIZE, 24, 24, 3)).astype(np.float32))
    viewpoints = [
        Viewpoint(
            centers=jnp.asarray(rng.uniform(-0.4, 0.4, (BATCH_SIZE, 2)).astype(np.float32)),
            scales=jnp.asarray(rng.uniform(0.3, 1.0, (BATCH_SIZE,)).astype(np.float32)),
        )
        for _ in range(3)
    ]
    return scenes, viewpoints, jnp.asarray(rng.integers(0, 7, BATCH_SIZE))


def _classifiers(config: CanViTConfig) -> dict[ComputeDtype, CanViTForImageClassification]:
    float32 = CanViTForImageClassification(canvit_config=config, n_classes=7, rngs=nnx.Rngs(0))
    bfloat16 = CanViTForImageClassification(canvit_config=config, n_classes=7, rngs=nnx.Rngs(1), compute_dtype="bfloat16")
    nnx.update(bfloat16, nnx.state(float32))
    return {"float32": float32, "bfloat16": bfloat16}


def _mean_cross_entropy(classifier: CanViTForImageClassification, seed: int):
    scenes, viewpoints, labels = _episode(seed)
    state = classifier.init_state(batch_size=BATCH_SIZE, canvas_grid_size=CANVAS_GRID_SIZE)
    all_logits = []
    for viewpoint in viewpoints:
        glimpse = sample_at_viewpoint(spatial=scenes, viewpoint=viewpoint, glimpse_size_px=16)
        logits, state = classifier(glimpse=glimpse, state=state, viewpoint=viewpoint)
        all_logits.append(logits)
    losses = [jax.nn.logsumexp(logits, axis=-1) - jnp.take_along_axis(logits, labels[:, None], -1)[:, 0] for logits in all_logits]
    return jnp.mean(jnp.stack(losses)), (all_logits[-1], state)


def test_bfloat16_rollout_stays_in_bfloat16_and_follows_float32(small_config: CanViTConfig) -> None:
    results = {name: _mean_cross_entropy(classifier, seed=0)[1] for name, classifier in _classifiers(small_config).items()}
    (float32_logits, float32_state), (bfloat16_logits, bfloat16_state) = results["float32"], results["bfloat16"]
    assert bfloat16_state.canvas.dtype == bfloat16_state.recurrent_cls.dtype == jnp.bfloat16
    assert bfloat16_logits.dtype == jnp.float32
    # Measured after three glimpses: 7e-3 for each.
    assert relative_l2(float32_logits, bfloat16_logits) < 2e-2
    assert relative_l2(float32_state.canvas, bfloat16_state.canvas) < 2e-2
    assert relative_l2(float32_state.recurrent_cls, bfloat16_state.recurrent_cls) < 2e-2


def test_bfloat16_gradients_reach_every_parameter_and_follow_float32(small_config: CanViTConfig) -> None:
    gradients = {}
    for name, classifier in _classifiers(small_config).items():
        grads = nnx.grad(lambda model: _mean_cross_entropy(model, seed=1)[0])(classifier)
        flat = nnx.to_flat_state(grads)
        gradients[name] = {".".join(map(str, path)): np.asarray(leaf[...]) for path, leaf in zip(flat.paths, flat.leaves)}
    assert gradients["bfloat16"].keys() == gradients["float32"].keys()
    for path, expected in gradients["float32"].items():
        actual = gradients["bfloat16"][path]
        assert actual.dtype == np.float32, path
        assert np.any(actual != 0), path
        # Measured over three glimpses: median 9e-3, worst 3.6e-2.
        assert relative_l2(expected, actual) < 1e-1, (path, relative_l2(expected, actual))
