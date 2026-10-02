"""CLI: the paper's datasets and figures, or named datasets, figures and diagrams.

  uv run python -m canvit_paper_exporter.run                 # the paper's datasets, then its figures
  uv run python -m canvit_paper_exporter.run list            # list names
  uv run python -m canvit_paper_exporter.run main_results ablation_loss
"""

import logging
import sys
import time

from canvit_paper_exporter import ablations, ade20k, bench, comparison, diagrams, flops, hf, in1k
from canvit_paper_exporter.core import (
    Dataset, Diagram, Figure, _fmt_elapsed, run_dataset, run_diagram, run_figure,
)

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
log = logging.getLogger(__name__)


DATASETS: list[Dataset] = [
    *ablations.DATASETS,
    *ade20k.DATASETS,
    *flops.DATASETS,
    in1k.frozen_dataset,
    in1k.finetuned_dataset,
    bench.dataset,
    hf.in1k_finetune_config_dataset,
    hf.dv3_probes_dataset,
]

# Run by name only.
NAMED_ONLY_DATASETS: list[Dataset] = ablations.DOWNSTREAM_DATASETS

FIGURES: list[Figure] = [
    *ablations.FIGURES,
    *ade20k.FIGURES,
    *flops.FIGURES,
    bench.figure,
    comparison.figure,
]

DIAGRAMS: list[Diagram] = diagrams.DIAGRAMS

_DS_BY_NAME = {d.name: d for d in [*DATASETS, *NAMED_ONLY_DATASETS]}
_FIG_BY_NAME = {f.name: f for f in FIGURES}
_DIAG_BY_NAME = {d.name: d for d in DIAGRAMS}


def _list() -> None:
    print("datasets:")
    for d in DATASETS:
        print(f"  {d.name}")
    print("datasets run by name only:")
    for d in NAMED_ONLY_DATASETS:
        print(f"  {d.name}")
    print("figures:")
    for f in FIGURES:
        print(f"  {f.name}")
    print("diagrams:")
    for d in DIAGRAMS:
        print(f"  {d.name}")


def _run_one(name: str) -> None:
    if name in _DS_BY_NAME:
        out = run_dataset(_DS_BY_NAME[name])
        print(f"  dataset: {out}")
    elif name in _FIG_BY_NAME:
        out = run_figure(_FIG_BY_NAME[name])
        print(f"  figure:  {out}")
    elif name in _DIAG_BY_NAME:
        out = run_diagram(_DIAG_BY_NAME[name])
        print(f"  diagram: {out}")
    else:
        available = ", ".join(sorted([*_DS_BY_NAME, *_FIG_BY_NAME, *_DIAG_BY_NAME]))
        raise SystemExit(f"Unknown name {name!r}. Available: {available}")


def main(argv: list[str]) -> None:
    if argv == ["list"]:
        _list()
        return
    t0 = time.perf_counter()
    if not argv:
        for d in DATASETS:
            run_dataset(d)
        for f in FIGURES:
            run_figure(f)
    else:
        for name in argv:
            _run_one(name)
    log.info("total wall-clock: %s", _fmt_elapsed(time.perf_counter() - t0))


if __name__ == "__main__":
    main(sys.argv[1:])
