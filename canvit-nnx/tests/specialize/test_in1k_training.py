import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from canvit_core import CanViTConfig
from flax import nnx

from canvit_nnx import CanViTForImageClassification
from canvit_nnx.policies import full_then_random
from canvit_nnx.specialize.in1k.data import PADDING_LABEL
from canvit_nnx.specialize.in1k.train import (
    Recipe,
    load_classifier,
    optimizer_transform,
    train_step,
    validation_step,
)
from canvit_nnx.training.run_directory import RunDirectory, load_optimizer_state

BATCH_SIZE, NUM_CLASSES, CANVAS_GRID_SIZE = 2, 7, 4
RECIPE = dataclasses.replace(Recipe(), batch_size=BATCH_SIZE, warmup_steps=2, peak_lr=1e-3, num_glimpses=2)


def _batch(step: int) -> tuple[jax.Array, jax.Array]:
    rng = np.random.default_rng(step)
    scenes = rng.normal(size=(BATCH_SIZE, 40, 40, 3)).astype(np.float32)
    return jnp.asarray(scenes), jnp.asarray(rng.integers(0, NUM_CLASSES, BATCH_SIZE))


def _step(classifier: CanViTForImageClassification, optimizer: nnx.Optimizer, step: int) -> float:
    viewpoints = full_then_random(jax.random.key(step), batch_size=BATCH_SIZE, num_glimpses=RECIPE.num_glimpses)
    results = train_step(
        classifier, optimizer, *_batch(step), viewpoints,
        label_smoothing=RECIPE.label_smoothing, canvas_grid_size=CANVAS_GRID_SIZE,
    )
    return float(results.loss)


def _parameters(classifier: CanViTForImageClassification) -> dict[str, np.ndarray]:
    flat = nnx.to_flat_state(nnx.state(classifier, nnx.Param))
    return {".".join(map(str, path)): np.asarray(leaf[...]) for path, leaf in zip(flat.paths, flat.leaves, strict=True)}


def test_resumed_training_continues_the_uninterrupted_run(small_config: CanViTConfig, tmp_path: Path) -> None:
    def classifier() -> CanViTForImageClassification:
        return CanViTForImageClassification(
            canvit_config=small_config, n_classes=NUM_CLASSES, rngs=nnx.Rngs(0), compute_dtype=RECIPE.compute_dtype
        )

    uninterrupted = classifier()
    optimizer = nnx.Optimizer(uninterrupted, optimizer_transform(RECIPE), wrt=nnx.Param)
    expected_losses = [_step(uninterrupted, optimizer, step) for step in range(4)]

    interrupted = classifier()
    optimizer = nnx.Optimizer(interrupted, optimizer_transform(RECIPE), wrt=nnx.Param)
    losses = [_step(interrupted, optimizer, step) for step in range(2)]
    run_directory = RunDirectory(tmp_path, is_leader=True)
    run_directory.save_checkpoint(model=interrupted, optimizer=optimizer)
    saved = run_directory.latest_checkpoint()
    assert saved is not None

    resumed = load_classifier(saved, compute_dtype=RECIPE.compute_dtype)
    optimizer = nnx.Optimizer(resumed, optimizer_transform(RECIPE), wrt=nnx.Param)
    load_optimizer_state(saved, optimizer)
    assert int(optimizer.step[...]) == 2
    losses += [_step(resumed, optimizer, step) for step in range(2, 4)]

    assert losses == expected_losses
    expected, actual = _parameters(uninterrupted), _parameters(resumed)
    assert expected.keys() == actual.keys()
    for name, value in expected.items():
        np.testing.assert_array_equal(actual[name], value, err_msg=name)


def test_validation_counts_scenes_and_ignores_padding(small_config: CanViTConfig) -> None:
    classifier = CanViTForImageClassification(canvit_config=small_config, n_classes=NUM_CLASSES, rngs=nnx.Rngs(0))
    scenes, labels = _batch(0)
    viewpoints = full_then_random(jax.random.key(0), batch_size=BATCH_SIZE + 1, num_glimpses=2)
    padded_scenes = jnp.concatenate((scenes, jnp.zeros_like(scenes[:1])))
    padded = validation_step(
        classifier, padded_scenes, jnp.append(labels, PADDING_LABEL), viewpoints, canvas_grid_size=CANVAS_GRID_SIZE
    )
    unpadded = validation_step(
        classifier, scenes, labels, tuple(viewpoint._replace(centers=viewpoint.centers[:BATCH_SIZE],
                                                             scales=viewpoint.scales[:BATCH_SIZE]) for viewpoint in viewpoints),
        canvas_grid_size=CANVAS_GRID_SIZE,
    )
    np.testing.assert_array_equal(padded.correct, unpadded.correct)
    np.testing.assert_allclose(padded.loss, unpadded.loss, rtol=1e-6)
    assert padded.correct.shape == (2,)
