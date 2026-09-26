"""python -m canvit_pytorch.pretrain: pretrain CanViT-B, or one of the paper's ablations with --ablation."""

import logging
from dataclasses import dataclass
from typing import Annotated

import tyro

from canvit_pytorch.pretrain.ablations import ABLATIONS, AblationSlug
from canvit_pytorch.pretrain.config import PretrainingConfig
from canvit_pytorch.pretrain.loop import train


@dataclass(frozen=True)
class Arguments:
    config: Annotated[PretrainingConfig, tyro.conf.arg(name="")]
    ablation: AblationSlug | None = None
    """Train this ablation of the paper: its change, on the ablations' shorter schedule."""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = tyro.cli(Arguments)
    config = args.config if args.ablation is None else ABLATIONS[args.ablation].configure(args.config)
    logging.getLogger(__name__).info(f"Config: {config}")
    train(config)


if __name__ == "__main__":
    main()
