"""Turn a trained probe `.pt` into a directory `from_pretrained` can load.

`canvit_pytorch.evaluate --probe-repo` resolves either an HF repo id or a local directory,
but a local directory only works if it holds a `save_pretrained` layout
(`config.json` + `model.safetensors`). A training run writes a single bare
`canvas_hidden_best_t9_miou<...>_step<...>.pt`, so pointing the evaluator at the
training directory fails.

`push_probes.py` solves this by uploading to the Hub. This does the same
materialization WITHOUT publishing, which is what a one-off internal evaluation
wants: same weights, same loader, no outward-facing action.

    uv run python scripts/specialize/materialize_probe.py \
        --probe  ~/projects/.../canvas_hidden_best_t9_miou0.3676_step39000.pt \
        --out-dir ~/scratch/probe_materialized/abl-baseline-c64

Weights come through `push_probes.load_probe`, which strict-loads and asserts no
missing or unexpected keys — so this cannot silently materialize a partial probe.
"""

from dataclasses import dataclass
from pathlib import Path

import tyro

from scripts.specialize.push_probes import load_probe


@dataclass
class Args:
    probe: Path
    """The trained `.pt` written by the probe training run."""
    out_dir: Path
    """Directory to write `config.json` + `model.safetensors` into."""


def main(args: Args) -> None:
    assert args.probe.exists(), f"Not found: {args.probe}"
    probe, meta = load_probe(args.probe)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    probe.save_pretrained(args.out_dir)
    written = sorted(p.name for p in args.out_dir.iterdir())
    print(f"{args.probe.name}\n  -> {args.out_dir}\n  wrote: {written}")
    # The step is in the source filename, not in the materialized directory, so
    # print it: two probes materialized from best-val checkpoints were selected
    # at DIFFERENT steps, and a comparison needs to say so.
    print(f"  source step/miou from filename: {args.probe.stem}")
    print(f"  feat_type: {meta.get('feat_type')}  config: {meta.get('config')}")


if __name__ == "__main__":
    main(tyro.cli(Args))
