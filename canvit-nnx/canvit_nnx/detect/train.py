"""COCO 2017 object detection: CanViT-B and a CenterNet head trained end to end in JAX/Flax NNX.

Each step rolls every scene out twice, under the F-IID and R-IID viewpoints
of pretraining with random viewpoints no smaller than the recipe's min_scale.
The CenterNet loss after every glimpse, averaged over glimpses and rollouts,
backpropagates through all of them. After global gradient-norm clipping,
AdamW trains the head and CanViT at their own learning rates and weight
decays, each warmed up linearly and then decayed along a cosine. Validation
scores COCO AP after each glimpse under C2F (canvit_nnx.detect.evaluate).
The batch is split over the devices of one host or of a multi-host slice.
"""

import contextlib
import logging
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

from canvit_nnx.detect.data import (
    NUM_CLASSES,
    NUM_TRAINING_IMAGES,
    TrainingBatch,
    annotation_file,
    normalize,
    read_split,
    training_batches,
)
from canvit_nnx.detect.evaluate import average_precision
from canvit_nnx.detect.loss import LossTerms, detection_loss, size_weight
from canvit_nnx.detect.model import CanViTForObjectDetection, load_detector
from canvit_nnx.detect.targets import Targets
from canvit_nnx.detect.visualize import rollout_panels
from canvit_nnx.model.config import ComputeDtype
from canvit_nnx.model.fused_canvas_attention import use_fused_kernels
from canvit_nnx.policies import all_random, full_then_random
from canvit_nnx.releases import (
    PRETRAINED,
    RELEASED_CANVAS_GRID_SIZE,
    RELEASED_GLIMPSE_SIZE_PX,
    RELEASED_SCENE_SIZE_PX,
)
from canvit_nnx.training.data_parallel import DataParallel
from canvit_nnx.training.loop import Cadence, train_until
from canvit_nnx.training.run_directory import MODEL, RunDirectory, load_optimizer_state
from canvit_nnx.training.utilization import StepCost
from canvit_nnx.viewpoint import Viewpoint, sample_at_viewpoint

log = logging.getLogger(__name__)

Array = jax.Array


@dataclass(frozen=True)
class Recipe:
    """What defines a detection run; the defaults are the paper's detector on the released CanViT-B.

    A resumed run keeps its recipe.
    """

    pretrained: str = PRETRAINED["in21k"]
    """The CanViT-B to start from: a Hub repository or a directory holding a CanViTForPretraining checkpoint."""
    compute_dtype: ComputeDtype = "bfloat16"
    batch_size: int = 1024
    """Scenes per optimizer step, split over the devices."""
    num_glimpses: int = 10
    """Glimpses of each training rollout."""
    min_scale: float = 0.25
    """Smallest scale of the random viewpoints; pretraining's MIN_SCALE gives glimpses too small to hold an object."""
    head_width: int = 256
    head_peak_lr: float = 4e-4
    canvit_peak_lr: float = 4e-4
    head_weight_decay: float = 0.1
    canvit_weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    """The global gradient norm is clipped to this."""
    epochs: int = 140
    warmup_fraction: float = 0.2
    """Of the steps, those of the linear warmup; the learning rates then decay to zero along a cosine."""
    canvas_grid_size: int = RELEASED_CANVAS_GRID_SIZE
    seed: int = 0
    """Seeds the head's initialization, the viewpoints and the data order."""

    @property
    def total_steps(self) -> int:
        return self.epochs * NUM_TRAINING_IMAGES // self.batch_size

    @property
    def warmup_steps(self) -> int:
        return int(self.total_steps * self.warmup_fraction)

    @property
    def size_weight(self) -> float:
        return size_weight(output_stride=RELEASED_SCENE_SIZE_PX / self.canvas_grid_size)


@dataclass(frozen=True)
class DetectionTrainingConfig:
    """Train CanViT-B with a CenterNet head for COCO 2017 object detection."""

    data_dir: Path
    """COCO 2017: annotations/instances_{train,val}2017.json, train2017/ and val2017/."""
    run_dir: Path
    """Where the recipe, metrics.jsonl, images and checkpoints go; a run whose run_dir holds a checkpoint resumes."""
    recipe: Recipe = Recipe()
    stop_after_step: int | None = None
    """End this invocation after that many optimizer steps, within the recipe's schedule; None: at its end."""
    checkpoint_every: int = 2000
    """Steps between checkpoints; a checkpoint is also saved when the invocation ends."""
    validate_every: int = 4000
    """Steps between validations; a validation also ends each invocation."""
    validation_glimpses: int | None = None
    """Glimpses of each validation rollout; None: the recipe's num_glimpses."""
    validation_repeats: int = 2
    validation_images: int | None = None
    """Validate on the first images only, by increasing id; None: all of val2017."""
    visualized_images: int = 4
    """Validation scenes drawn into the run directory's images, from the leader host's first batch."""
    log_every: int = 100
    decode_threads: int = 32
    multi_host: bool = False
    """Run on every host of a multi-host slice, each launched with the same command."""


class StepResults(NamedTuple):
    loss: Array  # the objective, mean over glimpses and rollouts
    terms: LossTerms  # its terms, same means
    grad_norm: Array  # before clipping


def learning_rate(peak: float, recipe: Recipe) -> optax.Schedule:
    return optax.warmup_cosine_decay_schedule(
        init_value=peak * 1e-6, peak_value=peak, warmup_steps=recipe.warmup_steps, decay_steps=recipe.total_steps
    )


def optimizer_transform(recipe: Recipe) -> optax.GradientTransformation:
    """Global-norm clipping, then AdamW on the head and on CanViT, each with its own learning rate and weight decay.

    Two AdamW, since scaling a group's gradients would leave Adam's updates unchanged.
    """
    return optax.chain(
        optax.clip_by_global_norm(recipe.max_grad_norm),
        optax.multi_transform(
            {
                "head": optax.adamw(learning_rate(recipe.head_peak_lr, recipe), weight_decay=recipe.head_weight_decay),
                "canvit": optax.adamw(
                    learning_rate(recipe.canvit_peak_lr, recipe), weight_decay=recipe.canvit_weight_decay
                ),
            },
            lambda parameters: jax.tree_util.tree_map_with_path(lambda path, _: path[0].key, parameters),
        ),
    )


def training_viewpoints(
    key: Array, *, batch_size: int, num_glimpses: int, min_scale: float
) -> tuple[tuple[Viewpoint, ...], tuple[Viewpoint, ...]]:
    """The F-IID and R-IID rollouts' viewpoints."""
    full_key, random_key = jax.random.split(key)
    return (
        full_then_random(full_key, batch_size=batch_size, num_glimpses=num_glimpses, min_scale=min_scale),
        all_random(random_key, batch_size=batch_size, num_glimpses=num_glimpses, min_scale=min_scale),
    )


draw_training_viewpoints = jax.jit(training_viewpoints, static_argnames=("batch_size", "num_glimpses", "min_scale"))


def rollout_loss(
    detector: CanViTForObjectDetection,
    scenes: Array,
    targets: Targets[Array],
    viewpoints: tuple[Viewpoint, ...],
    *,
    size_weight: float,
    canvas_grid_size: int,
) -> tuple[Array, LossTerms]:
    """The detection loss after each glimpse of [B, S, S, 3] normalized scenes, and its terms, averaged over glimpses."""
    state = detector.init_state(batch_size=scenes.shape[0], canvas_grid_size=canvas_grid_size)
    losses, terms = [], []
    for viewpoint in viewpoints:
        glimpse = sample_at_viewpoint(spatial=scenes, viewpoint=viewpoint, glimpse_size_px=RELEASED_GLIMPSE_SIZE_PX)
        maps, state = detector(glimpse=glimpse, state=state, viewpoint=viewpoint)
        loss, glimpse_terms = detection_loss(maps, targets, size_weight=size_weight)
        losses.append(loss)
        terms.append(glimpse_terms)
    return jnp.mean(jnp.stack(losses)), LossTerms(*(jnp.mean(jnp.stack(term)) for term in zip(*terms, strict=True)))


@nnx.jit(static_argnames=("size_weight", "canvas_grid_size"))
def train_step(
    detector: CanViTForObjectDetection,
    optimizer: nnx.Optimizer,
    scenes: Array,
    targets: Targets[Array],
    rollouts: tuple[tuple[Viewpoint, ...], ...],
    *,
    size_weight: float,
    canvas_grid_size: int,
) -> StepResults:
    def loss_function(detector: CanViTForObjectDetection) -> tuple[Array, LossTerms]:
        normalized = normalize(scenes)
        losses, terms = zip(
            *(
                rollout_loss(
                    detector, normalized, targets, viewpoints, size_weight=size_weight, canvas_grid_size=canvas_grid_size
                )
                for viewpoints in rollouts
            ),
            strict=True,
        )
        return jnp.mean(jnp.stack(losses)), LossTerms(*(jnp.mean(jnp.stack(term)) for term in zip(*terms, strict=True)))

    (loss, terms), grads = nnx.value_and_grad(loss_function, has_aux=True)(detector)
    optimizer.update(detector, grads)
    return StepResults(loss=loss, terms=terms, grad_norm=optax.tree.norm(grads))


class DetectionTraining:
    """A detection run's state in one invocation: model, optimizer, data, devices and run directory."""

    def __init__(self, config: DetectionTrainingConfig, data_parallel: DataParallel) -> None:
        self.config = config
        self.recipe = recipe = config.recipe
        self.data_parallel = data_parallel
        self.run_directory = RunDirectory(config.run_dir, is_leader=data_parallel.is_leader)
        self.training_key, self.validation_key = jax.random.split(jax.random.key(recipe.seed))
        self.learning_rates = {
            "head": learning_rate(recipe.head_peak_lr, recipe),
            "canvit": learning_rate(recipe.canvit_peak_lr, recipe),
        }
        self.run_directory.bind_recipe(recipe)
        self.training_split = read_split(config.data_dir, "train2017")
        num_images = len(self.training_split.images)
        assert num_images == NUM_TRAINING_IMAGES, f"train2017 has {num_images} images, not {NUM_TRAINING_IMAGES}"
        validation_split = read_split(config.data_dir, "val2017")
        self.validation_split = validation_split._replace(images=validation_split.images[: config.validation_images])
        saved = self.run_directory.latest_checkpoint()
        if saved is None:
            self.detector = CanViTForObjectDetection.from_pretrained_canvit(
                recipe.pretrained,
                num_classes=NUM_CLASSES,
                head_width=recipe.head_width,
                seed=recipe.seed,
                compute_dtype=recipe.compute_dtype,
            )
        else:
            self.detector = load_detector(saved / MODEL, compute_dtype=recipe.compute_dtype)
        self.optimizer = nnx.Optimizer(self.detector, optimizer_transform(recipe), wrt=nnx.Param)
        if saved is not None:
            load_optimizer_state(saved, self.optimizer)
        data_parallel.replicate(self.detector, self.optimizer)
        # Optimizer updates applied, counted on the host so that stepping never waits for the device.
        self.step = int(self.optimizer.step[...])
        log.info("%s at step %d", f"Resumed from {saved}" if saved else f"Initialized from {recipe.pretrained}", self.step)

    def batches(self) -> contextlib.closing[Generator[TrainingBatch]]:
        recipe = self.recipe
        return contextlib.closing(
            training_batches(
                self.training_split.images,
                batch_size=recipe.batch_size,
                rows=self.data_parallel.host_rows(recipe.batch_size),
                first_step=self.step,
                scene_size_px=RELEASED_SCENE_SIZE_PX,
                grid_size=recipe.canvas_grid_size,
                seed=recipe.seed,
                threads=self.config.decode_threads,
            )
        )

    def _inputs(self, batch: TrainingBatch) -> tuple[Array, Targets[Array], tuple[tuple[Viewpoint, ...], ...]]:
        """The batch's scenes and targets and this step's rollouts' viewpoints, as global arrays."""
        recipe, targets = self.recipe, batch.targets
        rollouts = draw_training_viewpoints(
            jax.random.fold_in(self.training_key, self.step),
            batch_size=recipe.batch_size,
            num_glimpses=recipe.num_glimpses,
            min_scale=recipe.min_scale,
        )
        scenes, heatmap, size, offset, mask = self.data_parallel.to_global(
            batch.scenes, targets.heatmap, targets.size, targets.offset, targets.mask
        )
        return (
            scenes,
            Targets(heatmap=heatmap, size=size, offset=offset, mask=mask),
            self.data_parallel.split_full_batch(rollouts, batch_size=recipe.batch_size),
        )

    def train_step(self, batch: TrainingBatch) -> StepResults:
        results = train_step(
            self.detector,
            self.optimizer,
            *self._inputs(batch),
            size_weight=self.recipe.size_weight,
            canvas_grid_size=self.recipe.canvas_grid_size,
        )
        self.step += 1
        return results

    def step_cost(self, batch: TrainingBatch) -> StepCost:
        lowered = train_step.lower(
            self.detector,
            self.optimizer,
            *self._inputs(batch),
            size_weight=self.recipe.size_weight,
            canvas_grid_size=self.recipe.canvas_grid_size,
        )
        return StepCost.of(lowered.compile())

    def summarize(self, window: list[StepResults]) -> dict[str, float]:
        """Means over the window's steps; the learning rates are those of the last step."""
        results = jax.device_get(window)
        metrics = {
            "train/loss": float(np.mean([result.loss for result in results])),
            "train/grad_norm": float(np.mean([result.grad_norm for result in results])),
        }
        for term in LossTerms._fields:
            metrics[f"train/loss_{term}"] = float(np.mean([getattr(result.terms, term) for result in results]))
        for group, schedule in self.learning_rates.items():
            metrics[f"train/{group}_learning_rate"] = float(np.asarray(schedule(self.step - 1)))
        return metrics

    def validate(self) -> dict[str, float]:
        start = time.perf_counter()
        evaluation = average_precision(
            self.detector,
            self.validation_split,
            annotations=annotation_file(self.config.data_dir, "val2017"),
            data_parallel=self.data_parallel,
            batch_size=self.recipe.batch_size,
            num_glimpses=self.config.validation_glimpses or self.recipe.num_glimpses,
            repeats=self.config.validation_repeats,
            key=self.validation_key,
            canvas_grid_size=self.recipe.canvas_grid_size,
            threads=self.config.decode_threads,
            num_rollouts=self.config.visualized_images,
        )
        for rollout in evaluation.rollouts:
            self.run_directory.save_image(
                step=self.step,
                name=f"image_{rollout.image.image_id}",
                image=rollout_panels(rollout, category_names=self.validation_split.category_names),
            )
        return evaluation.metrics | {"val/seconds": time.perf_counter() - start}

    def save_checkpoint(self) -> None:
        self.run_directory.save_checkpoint(model=self.detector, optimizer=self.optimizer)


def train(config: DetectionTrainingConfig) -> None:
    recipe = config.recipe
    data_parallel = DataParallel.start(multi_host=config.multi_host)
    last_step = recipe.total_steps if config.stop_after_step is None else min(config.stop_after_step, recipe.total_steps)
    log.info(
        "%d %s devices on %d hosts; %s compute%s; %d steps of %d scenes, %d glimpses per rollout",
        jax.device_count(),
        jax.default_backend(),
        jax.process_count(),
        recipe.compute_dtype,
        ", fused Canvas Attention kernels" if use_fused_kernels(jnp.dtype(recipe.compute_dtype)) else "",
        recipe.total_steps,
        recipe.batch_size,
        recipe.num_glimpses,
    )
    with jax.set_mesh(data_parallel.mesh):
        run = DetectionTraining(config, data_parallel)
        if run.step >= last_step:
            log.info("Nothing to do: step %d, asked to stop at %d", run.step, last_step)
            return
        with run.batches() as batches:
            train_until(
                run,
                batches,
                last_step=last_step,
                cadence=Cadence(
                    log_every=config.log_every,
                    validate_every=config.validate_every,
                    checkpoint_every=config.checkpoint_every,
                ),
                run_directory=run.run_directory,
                batch_size=recipe.batch_size,
            )


def main() -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
    train(tyro.cli(DetectionTrainingConfig, description=__doc__))


if __name__ == "__main__":
    main()
