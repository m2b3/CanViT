"""ImageNet-1k fine-tuning of CanViT-B in JAX/Flax NNX (paper, `supp:task-in1k`).

LP-FT: a pretrained CanViT-B, its CLS readout fused with DINOv3 ViT-B/16's
linear ImageNet-1k probe (CanViTForImageClassification.from_pretrained_with_probe),
then every parameter trained with AdamW. Each scene is seen through F-IID
glimpses, with a cross-entropy loss after every glimpse and backpropagation
through all of them. The defaults are the paper's recipe, which
canvit_pytorch.specialize.in1k_tpu also runs. The batch is split over the
devices of one host or of a multi-host slice.
"""

import contextlib
import functools
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

from canvit_nnx.model import CanViTForImageClassification
from canvit_nnx.model.config import ComputeDtype
from canvit_nnx.model.fused_canvas_attention import use_fused_kernels
from canvit_nnx.policies import full_then_random
from canvit_nnx.releases import (
    DINOV3_VITB16_IN1K_PROBE,
    PRETRAINED,
    RELEASED_CANVAS_GRID_SIZE,
    RELEASED_GLIMPSE_SIZE_PX,
    RELEASED_SCENE_SIZE_PX,
    PretrainingDataset,
)
from canvit_nnx.specialize.in1k.data import (
    NUM_TRAIN_IMAGES,
    NUM_VALIDATION_IMAGES,
    PADDING_LABEL,
    SceneBatch,
    training_batches,
    validation_batches,
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
    """What defines a fine-tuning run; the defaults are the paper's. A resumed run keeps its recipe."""

    pretraining_dataset: PretrainingDataset = "in21k"
    """The pretrained CanViT-B to fine-tune, named by its pretraining dataset."""
    compute_dtype: ComputeDtype = "bfloat16"
    batch_size: int = 256
    """Scenes per optimizer step, split over the devices."""
    peak_lr: float = 2.5e-5
    """The learning rate after linear warmup; it then decays to zero along a cosine."""
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    """The global gradient norm is clipped to this."""
    label_smoothing: float = 0.1
    warmup_steps: int = 25_000
    epochs: int = 20
    num_glimpses: int = 4
    canvas_grid_size: int = RELEASED_CANVAS_GRID_SIZE
    seed: int = 0
    """Seeds the viewpoints and the data order."""

    @property
    def steps_per_epoch(self) -> int:
        return NUM_TRAIN_IMAGES // self.batch_size

    @property
    def total_steps(self) -> int:
        return self.epochs * self.steps_per_epoch


@dataclass(frozen=True)
class FineTuningConfig:
    """Fine-tune CanViT-B for ImageNet-1k classification."""

    data_dir: Path
    """ImageNet-1k TFRecord shards, named train-* and validation-*."""
    run_dir: Path
    """Where the recipe, checkpoints and metrics.jsonl go; a run whose run_dir holds a checkpoint resumes from it."""
    recipe: Recipe = Recipe()
    stop_after_step: int | None = None
    """End this invocation after that many optimizer steps, within the recipe's schedule; None: at its end."""
    checkpoint_every: int = 1000
    """Steps between checkpoints; a checkpoint is also saved when the invocation ends."""
    log_every: int = 100
    decode_threads: int = 32
    multi_host: bool = False
    """Run on every host of a multi-host slice, each launched with the same command."""


class GlimpseTotals(NamedTuple):
    """Sums over the scenes of a batch, after each glimpse: [num_glimpses] each."""

    loss: Array
    correct: Array


class StepResults(NamedTuple):
    loss: Array  # the objective: label-smoothed cross-entropy, mean over glimpses and scenes
    glimpses: GlimpseTotals  # label-smoothed cross-entropy and correct predictions
    grad_norm: Array  # before clipping


def learning_rate(recipe: Recipe) -> optax.Schedule:
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=recipe.peak_lr,
        warmup_steps=recipe.warmup_steps,
        decay_steps=recipe.total_steps,
        end_value=0.0,
    )


def optimizer_transform(recipe: Recipe) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(recipe.max_grad_norm),
        optax.adamw(learning_rate(recipe), weight_decay=recipe.weight_decay),
    )


def episode_logits(
    classifier: CanViTForImageClassification, scenes: Array, viewpoints: tuple[Viewpoint, ...], *, canvas_grid_size: int
) -> Array:
    """[num_glimpses, B, num_classes] float32 logits after each glimpse of [B, S, S, 3] scenes."""
    state = classifier.init_state(batch_size=scenes.shape[0], canvas_grid_size=canvas_grid_size)
    logits = []
    for viewpoint in viewpoints:
        glimpse = sample_at_viewpoint(spatial=scenes, viewpoint=viewpoint, glimpse_size_px=RELEASED_GLIMPSE_SIZE_PX)
        glimpse_logits, state = classifier(glimpse=glimpse, state=state, viewpoint=viewpoint)
        logits.append(glimpse_logits)
    return jnp.stack(logits)


def lpft_loss(
    classifier: CanViTForImageClassification,
    scenes: Array,
    labels: Array,
    viewpoints: tuple[Viewpoint, ...],
    *,
    label_smoothing: float,
    canvas_grid_size: int,
) -> tuple[Array, GlimpseTotals]:
    """Label-smoothed cross-entropy after every glimpse, averaged over glimpses and scenes."""
    logits = episode_logits(classifier, scenes, viewpoints, canvas_grid_size=canvas_grid_size)
    targets = optax.smooth_labels(jax.nn.one_hot(labels, classifier.n_classes), label_smoothing)
    losses = optax.softmax_cross_entropy(logits, targets)
    correct = jnp.argmax(logits, axis=-1) == labels
    return losses.mean(), GlimpseTotals(loss=losses.sum(axis=1), correct=correct.sum(axis=1))


@nnx.jit(static_argnames=("label_smoothing", "canvas_grid_size"))
def train_step(
    classifier: CanViTForImageClassification,
    optimizer: nnx.Optimizer,
    scenes: Array,
    labels: Array,
    viewpoints: tuple[Viewpoint, ...],
    *,
    label_smoothing: float,
    canvas_grid_size: int,
) -> StepResults:
    loss_function = functools.partial(lpft_loss, label_smoothing=label_smoothing, canvas_grid_size=canvas_grid_size)
    (loss, glimpses), grads = nnx.value_and_grad(loss_function, has_aux=True)(classifier, scenes, labels, viewpoints)
    optimizer.update(classifier, grads)
    return StepResults(loss=loss, glimpses=glimpses, grad_norm=optax.tree.norm(grads))


@nnx.jit(static_argnames=("canvas_grid_size",))
def validation_step(
    classifier: CanViTForImageClassification,
    scenes: Array,
    labels: Array,
    viewpoints: tuple[Viewpoint, ...],
    *,
    canvas_grid_size: int,
) -> GlimpseTotals:
    """Cross-entropy and correct predictions summed over the batch's scenes, padding excluded."""
    logits = episode_logits(classifier, scenes, viewpoints, canvas_grid_size=canvas_grid_size)
    scene = labels != PADDING_LABEL
    losses = optax.softmax_cross_entropy_with_integer_labels(
        logits, jnp.broadcast_to(jnp.where(scene, labels, 0), logits.shape[:2])
    )
    correct = (jnp.argmax(logits, axis=-1) == labels) & scene
    return GlimpseTotals(loss=jnp.sum(losses * scene, axis=1), correct=correct.sum(axis=1))


draw_viewpoints = jax.jit(full_then_random, static_argnames=("batch_size", "num_glimpses"))


def load_classifier(checkpoint: Path, *, compute_dtype: ComputeDtype) -> CanViTForImageClassification:
    saved = CanViTForImageClassification.from_pretrained(checkpoint / MODEL)
    classifier = CanViTForImageClassification(
        canvit_config=saved.canvit.config, n_classes=saved.n_classes, rngs=nnx.Rngs(0), compute_dtype=compute_dtype
    )
    nnx.update(classifier, nnx.state(saved))
    return classifier


class FineTuning:
    """A fine-tuning run's state in one invocation: model, optimizer, devices and run directory."""

    def __init__(self, config: FineTuningConfig, data_parallel: DataParallel) -> None:
        self.config = config
        self.recipe = config.recipe
        self.data_parallel = data_parallel
        self.run_directory = RunDirectory(config.run_dir, is_leader=data_parallel.is_leader)
        self.training_key, self.validation_key = jax.random.split(jax.random.key(self.recipe.seed))
        self.learning_rate = learning_rate(self.recipe)
        self.run_directory.bind_recipe(self.recipe)
        saved = self.run_directory.latest_checkpoint()
        if saved is None:
            self.classifier = CanViTForImageClassification.from_pretrained_with_probe(
                pretrained_repo=PRETRAINED[self.recipe.pretraining_dataset],
                probe_repo=DINOV3_VITB16_IN1K_PROBE,
                compute_dtype=self.recipe.compute_dtype,
            )
        else:
            self.classifier = load_classifier(saved, compute_dtype=self.recipe.compute_dtype)
        self.optimizer = nnx.Optimizer(self.classifier, optimizer_transform(self.recipe), wrt=nnx.Param)
        if saved is not None:
            load_optimizer_state(saved, self.optimizer)
        data_parallel.replicate(self.classifier, self.optimizer)
        # Optimizer updates applied, counted on the host so that stepping never waits for the device.
        self.step = int(self.optimizer.step[...])
        log.info("%s at step %d", f"Resumed from {saved}" if saved else "Initialized from the pretrained model", self.step)

    def _inputs(self, batch: SceneBatch, key: Array) -> tuple[Array, Array, tuple[Viewpoint, ...]]:
        """The batch's scenes and labels and the viewpoints drawn from key, as global arrays."""
        viewpoints = draw_viewpoints(key, batch_size=self.recipe.batch_size, num_glimpses=self.recipe.num_glimpses)
        scenes, labels = self.data_parallel.to_global(batch.scenes, batch.labels)
        return scenes, labels, self.data_parallel.split_full_batch(viewpoints, batch_size=self.recipe.batch_size)

    def train_step(self, batch: SceneBatch) -> StepResults:
        inputs = self._inputs(batch, jax.random.fold_in(self.training_key, self.step))
        results = train_step(
            self.classifier,
            self.optimizer,
            *inputs,
            label_smoothing=self.recipe.label_smoothing,
            canvas_grid_size=self.recipe.canvas_grid_size,
        )
        self.step += 1
        return results

    def step_cost(self, batch: SceneBatch) -> StepCost:
        lowered = train_step.lower(
            self.classifier,
            self.optimizer,
            *self._inputs(batch, self.training_key),
            label_smoothing=self.recipe.label_smoothing,
            canvas_grid_size=self.recipe.canvas_grid_size,
        )
        return StepCost.of(lowered.compile())

    def summarize(self, window: list[StepResults]) -> dict[str, float]:
        """Means over the window's steps; the learning rate is that of the last step."""
        results = jax.device_get(window)
        num_scenes = len(results) * self.recipe.batch_size
        loss = np.sum([result.glimpses.loss for result in results], axis=0) / num_scenes
        correct = np.sum([result.glimpses.correct for result in results], axis=0) / num_scenes
        metrics = {
            "train/loss": float(np.mean([result.loss for result in results])),
            "train/grad_norm": float(np.mean([result.grad_norm for result in results])),
            "train/learning_rate": float(np.asarray(self.learning_rate(self.step - 1))),
        }
        for glimpse, (glimpse_loss, glimpse_accuracy) in enumerate(zip(loss, correct, strict=True)):
            metrics[f"train/loss_t{glimpse}"] = float(glimpse_loss)
            metrics[f"train/accuracy_t{glimpse}"] = float(glimpse_accuracy)
        return metrics

    def validate(self) -> dict[str, float]:
        """Accuracy and cross-entropy after each glimpse over the validation set; the same viewpoints every time."""
        start = time.perf_counter()
        batches = validation_batches(
            self.config.data_dir,
            num_images=NUM_VALIDATION_IMAGES,
            batch_size=self.recipe.batch_size,
            rows=self.data_parallel.host_rows(self.recipe.batch_size),
            scene_size_px=RELEASED_SCENE_SIZE_PX,
            threads=self.config.decode_threads,
        )
        totals = [
            validation_step(
                self.classifier,
                *self._inputs(batch, jax.random.fold_in(self.validation_key, index)),
                canvas_grid_size=self.recipe.canvas_grid_size,
            )
            for index, batch in enumerate(batches)
        ]
        loss = np.sum([np.asarray(total.loss) for total in totals], axis=0) / NUM_VALIDATION_IMAGES
        correct = np.sum([np.asarray(total.correct) for total in totals], axis=0) / NUM_VALIDATION_IMAGES
        metrics = {"val/scenes": NUM_VALIDATION_IMAGES, "val/seconds": time.perf_counter() - start}
        for glimpse in range(self.recipe.num_glimpses):
            metrics[f"val/accuracy_t{glimpse}"] = float(correct[glimpse])
            metrics[f"val/loss_t{glimpse}"] = float(loss[glimpse])
        return metrics

    def save_checkpoint(self) -> None:
        self.run_directory.save_checkpoint(model=self.classifier, optimizer=self.optimizer)


def train(config: FineTuningConfig) -> None:
    recipe = config.recipe
    data_parallel = DataParallel.start(multi_host=config.multi_host)
    rows = data_parallel.host_rows(recipe.batch_size)
    last_step = recipe.total_steps if config.stop_after_step is None else min(config.stop_after_step, recipe.total_steps)
    log.info(
        "%d %s devices on %d hosts; %s compute%s; %d steps of %d scenes, %d per epoch",
        jax.device_count(),
        jax.default_backend(),
        jax.process_count(),
        recipe.compute_dtype,
        ", fused Canvas Attention kernels" if use_fused_kernels(jnp.dtype(recipe.compute_dtype)) else "",
        recipe.total_steps,
        recipe.batch_size,
        recipe.steps_per_epoch,
    )
    with jax.set_mesh(data_parallel.mesh):
        run = FineTuning(config, data_parallel)
        if run.step >= last_step:
            log.info("Nothing to do: step %d, asked to stop at %d", run.step, last_step)
            return
        batches = training_batches(
            config.data_dir,
            batch_size=rows.stop - rows.start,
            scene_size_px=RELEASED_SCENE_SIZE_PX,
            seed=(recipe.seed, run.step),
            host=jax.process_index(),
            num_hosts=jax.process_count(),
            threads=config.decode_threads,
        )
        with contextlib.closing(batches):
            train_until(
                run,
                batches,
                last_step=last_step,
                cadence=Cadence(
                    log_every=config.log_every,
                    validate_every=recipe.steps_per_epoch,
                    checkpoint_every=config.checkpoint_every,
                ),
                run_directory=run.run_directory,
                batch_size=recipe.batch_size,
            )


def main() -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
    train(tyro.cli(FineTuningConfig, description=__doc__))


if __name__ == "__main__":
    main()
