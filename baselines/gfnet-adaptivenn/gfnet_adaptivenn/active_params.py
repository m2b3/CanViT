"""Parameters of the modules GFNet and AdaptiveNN run up to each glimpse, classifiers included.

    uv run python -m gfnet_adaptivenn.active_params --root <sources> --out active_params.json

A model's count after a glimpse sums `parameters()` of every module its
evaluation runs up to that glimpse, classifiers included; buffers are not
parameters, and modules that serve only training are not counted.
`gfnet.active_parameters` and `adaptivenn.active_parameters` name, from each
model's inference path, what each glimpse adds and what never runs. Both models
import a top-level package named `models`, so each is counted in its own process.
"""

import itertools
import json
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

DEFINITION = (
    "parameters() of the modules a model runs up to each glimpse, classifiers included; whole_checkpoint_M counts "
    "every tensor its checkpoint stores, buffers included, to identify the checkpoint"
)


def parameters_of(*modules: torch.nn.Module) -> set[torch.nn.Parameter]:
    return {parameter for module in modules for parameter in module.parameters()}


def cumulative_counts(
    added: Sequence[set[torch.nn.Parameter]], *, not_run: set[torch.nn.Parameter], every: set[torch.nn.Parameter],
) -> list[int]:
    """The parameters run up to each glimpse, from what each glimpse adds.

    Every parameter of the model is run at a single glimpse or named as never run.
    """
    run = set().union(*added)
    assert all(a.isdisjoint(b) for a, b in itertools.combinations(added, 2)), "a parameter is added at two glimpses"
    assert run.isdisjoint(not_run), "a parameter is both run and not run"
    unaccounted = every - run - not_run
    assert not unaccounted, f"{len(unaccounted)} parameter tensors are neither run nor named as not run"
    assert run | not_run <= every, "a counted parameter is not the model's"
    return list(itertools.accumulate(sum(parameter.numel() for parameter in step) for step in added))


def millions(count: int) -> float:
    return round(count / 1e6, 3)


@dataclass(frozen=True)
class ActiveParams:
    root: Path
    out: Path


if __name__ == "__main__":
    command = tyro.cli(ActiveParams)
    command.out.parent.mkdir(parents=True, exist_ok=True)
    parts = {}
    with tempfile.TemporaryDirectory() as directory:
        for baseline in ("gfnet", "adaptivenn"):
            part = Path(directory) / f"{baseline}.json"
            subprocess.run((sys.executable, "-m", f"gfnet_adaptivenn.{baseline}", "parameters",
                            "--root", str(command.root), "--out", str(part)), check=True)
            parts[baseline] = json.loads(part.read_text())
    result = {
        "definition": DEFINITION,
        "provenance": {baseline: part["provenance"] for baseline, part in parts.items()},
        "adaptivenn_deit_s": parts["adaptivenn"]["model"],
        "gfnet": parts["gfnet"]["models"],
    }
    command.out.write_text(json.dumps(result, indent=1) + "\n")
