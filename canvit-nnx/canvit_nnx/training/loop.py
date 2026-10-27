"""The training loop: optimizer steps, metric windows, validations and checkpoints."""

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import jax

from canvit_nnx.training.run_directory import RunDirectory
from canvit_nnx.training.utilization import StepCost


class Trainer[Batch, Results](Protocol):
    """A run's model, optimizer and data-parallel placement, stepped by train_until."""

    @property
    def step(self) -> int:
        """Optimizer updates applied so far."""
        ...

    def train_step(self, batch: Batch) -> Results:
        """Apply an optimizer update; the results may still be on the device."""
        ...

    def step_cost(self, batch: Batch) -> StepCost:
        """XLA's cost of the compiled training step on a batch of this shape."""
        ...

    def summarize(self, window: list[Results]) -> dict[str, float]:
        """Metrics over consecutive steps, the last of which is the current one."""
        ...

    def validate(self) -> dict[str, float]: ...

    def save_checkpoint(self) -> None: ...


@dataclass(frozen=True)
class Cadence:
    """Steps between metric windows, validations and checkpoints; the last step of an invocation has all three."""

    log_every: int
    validate_every: int
    checkpoint_every: int


def train_until[Batch, Results](
    trainer: Trainer[Batch, Results],
    batches: Iterator[Batch],
    *,
    last_step: int,
    cadence: Cadence,
    run_directory: RunDirectory,
    batch_size: int,
) -> None:
    """Step until last_step.

    The first step of an invocation is a window of its own, without throughput:
    it includes compilation. The step's cost is then measured, which compiles
    the step once more, outside any window. Validations and checkpoints are
    left out of the windows' time.
    """
    first_step = trainer.step + 1
    cost: StepCost | None = None
    window: list[Results] = []
    window_start = time.perf_counter()
    while trainer.step < last_step:
        batch = next(batches)
        window.append(trainer.train_step(batch))
        step = trainer.step
        if step == first_step or step % cadence.log_every == 0 or step == last_step:
            jax.block_until_ready(window)
            seconds_per_step = (time.perf_counter() - window_start) / len(window)
            metrics = trainer.summarize(window)
            if cost is None:
                cost = trainer.step_cost(batch)
            else:
                metrics["train/scenes_per_second"] = batch_size / seconds_per_step
                metrics |= {f"train/{name}": value for name, value in cost.utilization(seconds_per_step).items()}
            run_directory.record_metrics(step=step, metrics=metrics)
            window, window_start = [], time.perf_counter()
        pause_start = time.perf_counter()
        if step % cadence.validate_every == 0 or step == last_step:
            run_directory.record_metrics(step=step, metrics=trainer.validate())
        if step % cadence.checkpoint_every == 0 or step == last_step:
            trainer.save_checkpoint()
        window_start += time.perf_counter() - pause_start
