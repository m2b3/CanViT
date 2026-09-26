"""Push ablation checkpoints to HuggingFace Hub.

Naming: {owner}/canvitb16-abl-{slug}-{YYYY-MM-DD} (date from checkpoint
timestamp, slug from the registry below).

Usage:
    uv run python scripts/push_ablation_checkpoints.py --ckpt-dir <path> --dry-run
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import tyro
from canvit_pytorch.model.pretraining.hub import (
    descriptive_metadata,
    reconstruct_pretrain_model,
    upload_to_hf,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ckpt_dir stem → short slug for HF repo name.
_SLUG: dict[str, str] = {
    "abl-baseline-200k": "baseline",
    "abl-qkvo-dcan256-200k": "qkvo-dcan256",
    "abl-qkvo-dcan384-200k": "qkvo-dcan384",
    "abl-dcan256-200k": "dcan256",
    "abl-no-dense-200k": "no-dense",
    "abl-no-fiid-200k": "no-fiid-1riid",
    "abl-2riid-no-fiid-200k": "no-fiid-2riid",
    "abl-no-bptt-200k": "no-bptt",
    "abl-no-reads-200k": "no-reads",
    "abl-no-vpe-200k": "no-vpe",
    "abl-rw-stride6-200k": "rw-stride6",
    "abl-vit-s-200k": "vit-s",
}


@dataclass
class Args:
    ckpt_dir: Path
    owner: str = "canvit"
    dry_run: bool = False


def main(args: Args) -> None:
    assert args.ckpt_dir.is_dir(), f"Not a directory: {args.ckpt_dir}"
    files = sorted(args.ckpt_dir.glob("*.pt"))
    assert len(files) > 0, f"No .pt files in {args.ckpt_dir}"

    log.info("%s %d checkpoints from %s",
             "DRY RUN:" if args.dry_run else "Pushing", len(files), args.ckpt_dir)

    for f in files:
        stem = f.stem
        slug = _SLUG.get(stem)
        assert slug is not None, (
            f"Unknown checkpoint '{stem}' — not in _SLUG. Known: {sorted(_SLUG)}"
        )

        raw = torch.load(f, map_location="cpu", weights_only=False)
        ts = datetime.fromisoformat(raw["timestamp"])
        repo_id = f"{args.owner}/canvitb16-abl-{slug}-{ts:%Y-%m-%d}"
        log.info("  %s → %s (step=%d, %s)", stem, repo_id, raw["step"], ts.date())

        if args.dry_run:
            continue

        model = reconstruct_pretrain_model(raw)
        upload_to_hf(model, repo_id, private=True, extra_metadata=descriptive_metadata(raw))
        del model, raw
        torch.cuda.empty_cache()

    log.info("Done.")


if __name__ == "__main__":
    main(tyro.cli(Args))
