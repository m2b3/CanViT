import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from canvit_core import CanViTConfig
from flax import nnx

from canvit_nnx.detect.data import NUM_CLASSES, annotation_file, read_split
from canvit_nnx.detect.evaluate import average_precision
from canvit_nnx.detect.model import CanViTForObjectDetection, load_detector
from canvit_nnx.detect.targets import Targets, render
from canvit_nnx.detect.train import Recipe, optimizer_transform, train_step, training_viewpoints
from canvit_nnx.detect.visualize import rollout_panels
from canvit_nnx.releases import RELEASED_SCENE_SIZE_PX
from canvit_nnx.training.data_parallel import DataParallel
from canvit_nnx.training.run_directory import MODEL, RunDirectory, load_optimizer_state

BATCH_SIZE, CANVAS_GRID_SIZE = 2, 4
RECIPE = dataclasses.replace(Recipe(), batch_size=BATCH_SIZE, num_glimpses=2, canvas_grid_size=CANVAS_GRID_SIZE)


def _detector(config: CanViTConfig) -> CanViTForObjectDetection:
    return CanViTForObjectDetection(canvit_config=config, num_classes=NUM_CLASSES, head_width=8, rngs=nnx.Rngs(0))


def _batch(step: int) -> tuple[jax.Array, Targets[jax.Array]]:
    rng = np.random.default_rng(step)
    scenes = rng.integers(0, 256, (BATCH_SIZE, 64, 64, 3), dtype=np.uint8)
    corners = rng.uniform(0, 0.5, (BATCH_SIZE, 3, 2))
    boxes = np.concatenate([corners, corners + rng.uniform(0.1, 0.5, (BATCH_SIZE, 3, 2))], axis=-1).astype(np.float32)
    targets = [
        render(scene_boxes, rng.integers(0, NUM_CLASSES, 3), grid_size=CANVAS_GRID_SIZE, num_classes=NUM_CLASSES)
        for scene_boxes in boxes
    ]
    return jnp.asarray(scenes), Targets(*(jnp.asarray(np.stack(field)) for field in zip(*targets, strict=True)))


def _step(detector: CanViTForObjectDetection, optimizer: nnx.Optimizer, step: int, recipe: Recipe = RECIPE) -> float:
    rollouts = training_viewpoints(
        jax.random.key(step), batch_size=BATCH_SIZE, num_glimpses=recipe.num_glimpses, min_scale=recipe.min_scale
    )
    results = train_step(
        detector, optimizer, *_batch(step), rollouts, size_weight=recipe.size_weight, canvas_grid_size=CANVAS_GRID_SIZE
    )
    return float(results.loss)


def _parameters(node: nnx.Module) -> dict[str, np.ndarray]:
    flat = nnx.to_flat_state(nnx.state(node, nnx.Param))
    return {".".join(map(str, path)): np.asarray(leaf[...]) for path, leaf in zip(flat.paths, flat.leaves, strict=True)}


def test_the_head_and_canvit_train_at_their_own_learning_rates(small_config: CanViTConfig) -> None:
    frozen_head = dataclasses.replace(RECIPE, head_peak_lr=0.0)
    detector = _detector(small_config)
    before = _parameters(detector)
    _step(detector, nnx.Optimizer(detector, optimizer_transform(frozen_head), wrt=nnx.Param), 0, frozen_head)
    after = _parameters(detector)
    changed = {name for name in before if not np.array_equal(before[name], after[name])}
    assert changed and all(name.startswith("canvit.") for name in changed), sorted(changed)


def test_a_resumed_run_continues_the_uninterrupted_run(small_config: CanViTConfig, tmp_path: Path) -> None:
    uninterrupted = _detector(small_config)
    optimizer = nnx.Optimizer(uninterrupted, optimizer_transform(RECIPE), wrt=nnx.Param)
    expected_losses = [_step(uninterrupted, optimizer, step) for step in range(4)]

    interrupted = _detector(small_config)
    optimizer = nnx.Optimizer(interrupted, optimizer_transform(RECIPE), wrt=nnx.Param)
    losses = [_step(interrupted, optimizer, step) for step in range(2)]
    run_directory = RunDirectory(tmp_path, is_leader=True)
    run_directory.save_checkpoint(model=interrupted, optimizer=optimizer)
    saved = run_directory.latest_checkpoint()
    assert saved is not None

    resumed = load_detector(saved / MODEL, compute_dtype="float32")
    optimizer = nnx.Optimizer(resumed, optimizer_transform(RECIPE), wrt=nnx.Param)
    load_optimizer_state(saved, optimizer)
    losses += [_step(resumed, optimizer, step) for step in range(2, 4)]

    assert losses == expected_losses
    assert all(np.isfinite(losses))
    expected, actual = nnx.to_flat_state(nnx.state(uninterrupted)), nnx.to_flat_state(nnx.state(resumed))
    for path, value, resumed_value in zip(expected.paths, expected.leaves, actual.leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(resumed_value[...]), np.asarray(value[...]), err_msg=str(path))


def test_evaluation_scores_and_draws_every_glimpse(small_config: CanViTConfig, coco_dir: Path) -> None:
    split = read_split(coco_dir, "val2017")
    detector = _detector(small_config)
    # Boxes of a cell everywhere: an untrained head's sizes are as often negative, and those boxes are not scored.
    size = detector.head.size
    assert size.bias is not None
    size.kernel[...] = jnp.zeros_like(size.kernel[...])
    size.bias[...] = jnp.ones_like(size.bias[...])
    data_parallel = DataParallel.start(multi_host=False)
    with jax.set_mesh(data_parallel.mesh):
        evaluation = average_precision(
            detector,
            split,
            annotations=annotation_file(coco_dir, "val2017"),
            data_parallel=data_parallel,
            batch_size=4,
            num_glimpses=2,
            repeats=2,
            key=jax.random.key(0),
            canvas_grid_size=CANVAS_GRID_SIZE,
            threads=2,
            num_rollouts=3,
        )
    for glimpse in range(2):
        assert 0.0 <= evaluation.metrics[f"val/AP_t{glimpse}"] <= 1.0
        assert evaluation.metrics[f"val/AP_t{glimpse}_ci_low"] <= evaluation.metrics[f"val/AP_t{glimpse}_ci_high"]
    assert evaluation.metrics["val/images"] == len(split.images)
    assert [rollout.image.image_id for rollout in evaluation.rollouts] == [image.image_id for image in split.images[:3]]
    panels = rollout_panels(evaluation.rollouts[0], category_names=split.category_names)
    assert panels.size == (2 * RELEASED_SCENE_SIZE_PX, RELEASED_SCENE_SIZE_PX)
