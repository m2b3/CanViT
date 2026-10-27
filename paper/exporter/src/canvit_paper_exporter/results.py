"""Evaluation results in canvit_pytorch.evaluate's layout: torch files whose metadata records the configuration of the
run. Readers select and group results by that configuration, never by file name."""

from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import NamedTuple

import torch
from canvit_pytorch.hub.repos import RELEASED_GLIMPSE_SIZE_PX

from canvit_paper_exporter.paths import Inputs

Result = dict


def load_result(path: Path) -> Result:
    return torch.load(path, map_location="cpu", weights_only=True)


def results_in(entry: str) -> Inputs:
    """Every result in a data/ entry."""
    return Inputs(entry, ("*.pt",))


def load_results(inputs: Inputs) -> dict[str, Result]:
    """Every result inputs selects by file name, in file-name order: the order bootstrap CIs resample runs in."""
    return {f.name: load_result(f) for f in inputs.files()}


def config(result: Result) -> dict:
    return result["metadata"]["config"]


class Viewing(NamedTuple):
    """How a CanViT evaluation viewed its scenes."""

    policy: str
    scene_size: int
    canvas_grid: int


def viewing(result: Result) -> Viewing:
    """The run's viewing; every result the paper reports took the released 128 px glimpses its FLOP counts assume."""
    c = config(result)
    episode = c["episode"]
    assert episode["glimpse_size_px"] == RELEASED_GLIMPSE_SIZE_PX, episode
    return Viewing(episode["policy"], c["scene_size_px"], episode["canvas_grid_size"])


def group_by[T, K](items: Iterable[T], key: Callable[[T], K]) -> dict[K, list[T]]:
    """Items by key, each group in the items' order."""
    groups: dict[K, list[T]] = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    return dict(groups)


def miou_curve(result: Result) -> list[float]:
    """ADE20K mIoU after each glimpse."""
    mious = result["mious"]
    return [mious[f"t{t}"] for t in range(len(mious))]


def top1_curve(result: Result) -> list[float]:
    """ImageNet-1k top-1 accuracy after each glimpse, from the recorded top-5 predictions."""
    return (result["top_k_preds"][:, :, 0] == result["labels"][:, None]).float().mean(dim=0).tolist()
