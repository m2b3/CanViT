"""ADE20K probe training (CanViT canvas probes + DINOv3 baseline probes).

Usage:
    uv run python -m canvit_specialize.training.ade20k train ...
    uv run python -m canvit_specialize.training.ade20k train-dinov3-probe ...
"""

from typing import Annotated, Union

import tyro

from canvit_specialize.training.ade20k.train_dinov3 import DINOv3ProbeTrainConfig
from canvit_specialize.training.ade20k.train_dinov3 import train as run_train_dinov3
from canvit_specialize.training.ade20k.train_canvit import train as run_train
from canvit_specialize.training.ade20k.config import Config as TrainConfig

# tyro subcommand unions: valid at runtime, but basedpyright can't resolve
# Union[Annotated[...]] as TypeForm.
_Command = Union[
    Annotated[TrainConfig, tyro.conf.subcommand("train")],
    Annotated[DINOv3ProbeTrainConfig, tyro.conf.subcommand("train-dinov3-probe")],
]


def main() -> None:
    cmd: TrainConfig | DINOv3ProbeTrainConfig = tyro.cli(_Command)  # pyright: ignore[reportCallIssue,reportArgumentType]
    match cmd:
        case TrainConfig():
            run_train(cmd)
        case DINOv3ProbeTrainConfig():
            run_train_dinov3(cmd)


if __name__ == "__main__":
    main()
