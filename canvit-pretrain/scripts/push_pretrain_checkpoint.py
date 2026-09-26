"""Push one pretraining checkpoint to the HuggingFace Hub.

Reconstructs the model, attaches anonymization-safe descriptive metadata, and
generates a model card. Repo-ids for the main checkpoints live in
canvit_pytorch.checkpoints.PRETRAIN_CHECKPOINTS.

Usage:
    uv run python scripts/push_pretrain_checkpoint.py \
        --ckpt <path>/step-2001792.pt \
        --repo-id canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in1k-dv3b16-2026-06-22 \
        --public
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import tyro
from canvit_pytorch.model.pretraining.hub import push_checkpoint_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


@dataclass
class Args:
    ckpt: Path
    repo_id: str
    public: bool = False
    card: bool = True


def main(args: Args) -> None:
    repo = push_checkpoint_file(
        args.ckpt, args.repo_id,
        private=not args.public, with_card=args.card,
    )
    logging.info("Pushed: https://huggingface.co/%s", repo)


if __name__ == "__main__":
    main(tyro.cli(Args))
