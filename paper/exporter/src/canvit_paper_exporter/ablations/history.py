"""The ablation runs' pretraining histories, exported from Comet: one JSON per run, {metric: [point, ...]}.

A resumed run logged the steps since its last checkpoint again, so a step can hold several points. The point
logged last belongs to the run that continued to the released checkpoint; each point's logging timestamp says
which that is.
"""

import json
from typing import NamedTuple

import numpy as np

from canvit_paper_exporter.paths import Inputs


class Series(NamedTuple):
    """A metric's value at each logged step, steps increasing."""

    steps: np.ndarray
    values: np.ndarray


def _latest_per_step(points: list[dict], where: str) -> Series:
    if not points:
        return Series(np.empty(0, dtype=np.int64), np.empty(0))
    assert all("timestamp" in p for p in points), (
        f"{where}: points lack logging timestamps; export the history with each point's timestamp"
    )
    steps = np.array([p["step"] for p in points], dtype=np.int64)
    times = np.array([p["timestamp"] for p in points], dtype=np.int64)
    values = np.array([p["value"] for p in points], dtype=np.float64)
    order = np.lexsort((times, steps))
    steps, times, values = steps[order], times[order], values[order]
    same_step = steps[1:] == steps[:-1]
    tied = same_step & (times[1:] == times[:-1])
    assert not tied.any(), f"{where}: step {steps[1:][tied][0]} has two points logged at the same time"
    last = np.r_[~same_step, True]
    return Series(steps[last], values[last])


def _file(slug: str) -> str:
    return f"{slug}.json"


def runs(*slugs: str) -> Inputs:
    """The histories of these ablation runs."""
    return Inputs("comet", tuple(map(_file, slugs)))


def load(histories: Inputs, slug: str) -> dict[str, Series]:
    """Every metric of an ablation run's history, keeping the last-logged value of each step."""
    path = histories.file(_file(slug))
    return {name: _latest_per_step(points, f"{path}: {name}") for name, points in json.loads(path.read_text()).items()}
