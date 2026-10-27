"""CLI: the paper's datasets and figures, or named datasets, figures and diagrams.

  uv run python -m canvit_paper_exporter.run                 # the paper's datasets, then its figures
  uv run python -m canvit_paper_exporter.run list            # list names
  uv run python -m canvit_paper_exporter.run inputs          # what each reads under data/ and writes, as JSON
  uv run python -m canvit_paper_exporter.run main_results ablation_loss
"""

import inspect
import json
import logging
import sys
import time

from canvit_paper_exporter import (
    ablations, ade20k, architecture_control, bench, coco_detection, dataset_scale, diagrams, flops, hf, in1k,
    in1k_efficiency, learned_policies, main_results, policy_styles,
)
from canvit_paper_exporter.core import (
    Dataset, Diagram, Figure, _fmt_elapsed, run_dataset, run_diagram, run_figure,
)
from canvit_paper_exporter.paths import EXPORTS, Inputs

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
    dataset_scale.dataset,
    architecture_control.dataset,
    coco_detection.dataset,
    in1k_efficiency.dataset,
    learned_policies.dataset,
    policy_styles.dataset,
]

FIGURES: list[Figure] = [
    *ablations.FIGURES,
    *ade20k.FIGURES,
    *flops.FIGURES,
    bench.figure,
    main_results.figure,
    learned_policies.figure,
]

DIAGRAMS: list[Diagram] = diagrams.DIAGRAMS

_DS_BY_NAME = {d.name: d for d in DATASETS}
_FIG_BY_NAME = {f.name: f for f in FIGURES}
_DIAG_BY_NAME = {d.name: d for d in DIAGRAMS}


def _list() -> None:
    print("datasets:")
    for d in DATASETS:
        print(f"  {d.name}")
    print("figures:")
    for f in FIGURES:
        print(f"  {f.name}")
    print("diagrams:")
    for d in DIAGRAMS:
        print(f"  {d.name}")


def _about(f: object) -> str | None:
    """The first paragraph of the docstring of the module that defines f."""
    doc = inspect.getmodule(f).__doc__
    return " ".join(doc.split("\n\n")[0].split()) if doc else None


def _inputs() -> None:
    """Print, as JSON, every dataset and figure of the default run with the data/ files it reads and the exports it
    writes, then every diagram with its exports; file paths are relative to data/<entry>/ and exports/."""
    def reads(inputs: tuple[Inputs, ...]) -> list[dict]:
        return [
            {"entry": i.entry, "patterns": list(i.patterns),
             "files": [f.relative_to(i.directory).as_posix() for f in i.files()]}
            for i in inputs
        ]

    def writes(outputs: tuple) -> list[str]:
        return [out.relative_to(EXPORTS).as_posix() for out in outputs]

    print(json.dumps({
        "datasets": [
            {"name": d.name, "about": _about(d.compute), "exports": writes(d.outputs), "inputs": reads(d.inputs)}
            for d in DATASETS
        ],
        "figures": [
            {"name": f.name, "about": _about(f.plot), "exports": writes(f.outputs),
             "datasets": [d.name for d in f.datasets], "inputs": reads(f.inputs)}
            for f in FIGURES
        ],
        "diagrams": [{"name": d.name, "about": _about(d.prep), "exports": writes(d.outputs)} for d in DIAGRAMS],
    }, indent=1))


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
    if argv == ["inputs"]:
        _inputs()
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
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    # matplotlib subsets the figures' font with fontTools, which logs each font table it prunes; the Hub client's
    # HTTP library logs each request.
    for chatty in ("fontTools", "httpx"):
        logging.getLogger(chatty).setLevel(logging.WARNING)
    main(sys.argv[1:])
