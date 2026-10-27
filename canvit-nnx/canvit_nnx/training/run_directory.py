"""A training run's directory: the recipe it runs, its metrics, images and checkpoints.

- recipe.json: the recipe, a flat dataclass; a later invocation with another
  recipe refuses to run.
- metrics.jsonl: one JSON object per record, with the number of optimizer steps taken.
- images/step_<steps>/<name>.png: pictures of the model's outputs, such as predictions at validation.
- checkpoints/step_<steps>/: model/ (config.json, model.safetensors) and
  optimizer.safetensors, whose step counts the optimizer updates applied.
  checkpoints/latest names the newest complete checkpoint and is replaced
  atomically after it is written; every other checkpoint is then removed.

On a multi-host slice the directory must be one that every host sees, such as
a mounted bucket.
"""

import dataclasses
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import flax.nnx as nnx
from PIL import Image

from canvit_nnx.hub import HubMixin, load_state, save_state

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

log = logging.getLogger(__name__)

MODEL = "model"
"""The subdirectory of a checkpoint that holds the model, loadable with its class's from_pretrained."""


@dataclass(frozen=True)
class RunDirectory:
    """A run directory as one host sees it: every host reads, only the leader writes."""

    path: Path
    is_leader: bool

    @property
    def checkpoints(self) -> Path:
        return self.path / "checkpoints"

    def bind_recipe(self, recipe: "DataclassInstance") -> None:
        """Record the recipe in a new run directory, or check that an existing one ran with it."""
        path = self.path / "recipe.json"
        if path.exists():
            recorded = type(recipe)(**json.loads(path.read_text()))
            if recorded != recipe:
                raise ValueError(f"{path} records {recorded}; this invocation asks for {recipe}")
        elif self.is_leader:
            self.path.mkdir(parents=True, exist_ok=True)
            # Renamed into place once written, since the other hosts check for it at the same moment.
            staged = self.path / ".recipe.json.partial"
            staged.write_text(json.dumps(dataclasses.asdict(recipe), indent=2) + "\n")
            os.replace(staged, path)

    def record_metrics(self, *, step: int, metrics: dict[str, float]) -> None:
        if not self.is_leader:
            return
        log.info("step %d | %s", step, " ".join(f"{name}={value:.4g}" for name, value in metrics.items()))
        with (self.path / "metrics.jsonl").open("a") as file:
            file.write(json.dumps({"step": step} | metrics) + "\n")

    def save_image(self, *, step: int, name: str, image: Image.Image) -> None:
        if not self.is_leader:
            return
        directory = self.path / "images" / f"step_{step:07d}"
        directory.mkdir(parents=True, exist_ok=True)
        image.save(directory / f"{name}.png")

    def save_checkpoint(self, *, model: HubMixin, optimizer: nnx.Optimizer) -> None:
        if not self.is_leader:
            return
        checkpoint = self.checkpoints / f"step_{int(optimizer.step[...]):07d}"
        assert checkpoint != self.latest_checkpoint(), f"{checkpoint} is already the latest checkpoint"
        staged = self.checkpoints / f".{checkpoint.name}.partial"
        # A checkpoint at this step that latest does not name was left by an invocation stopped before naming it.
        for leftover in (staged, checkpoint):
            shutil.rmtree(leftover, ignore_errors=True)
        staged.mkdir(parents=True)
        model.save_pretrained(staged / MODEL)
        save_state(optimizer, staged / "optimizer.safetensors")
        os.replace(staged, checkpoint)
        pointer = self.checkpoints / ".latest.partial"
        pointer.write_text(checkpoint.name)
        os.replace(pointer, self.checkpoints / "latest")
        for other in self.checkpoints.glob("step_*"):
            if other != checkpoint:
                shutil.rmtree(other)
        log.info("Saved %s", checkpoint)

    def latest_checkpoint(self) -> Path | None:
        pointer = self.checkpoints / "latest"
        return self.checkpoints / pointer.read_text() if pointer.exists() else None


def load_optimizer_state(checkpoint: Path, optimizer: nnx.Optimizer) -> None:
    load_state(optimizer, checkpoint / "optimizer.safetensors")
