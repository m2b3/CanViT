import logging

import tyro

from canvit_pytorch.specialize.ade20k.canvas_probe import CanvasProbeConfig
from canvit_pytorch.specialize.ade20k.dinov3_probe import DINOv3ProbeConfig
from canvit_pytorch.specialize.ade20k.train import train_probe


def main() -> None:
    """Train a linear ADE20K probe on frozen features; print the run directory."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    setup: CanvasProbeConfig | DINOv3ProbeConfig = tyro.extras.subcommand_cli_from_dict(
        {"canvas-probe": CanvasProbeConfig, "dinov3-probe": DINOv3ProbeConfig},
        description=main.__doc__,
        config=(tyro.conf.OmitArgPrefixes,),
    )
    print(train_probe(setup))


if __name__ == "__main__":
    main()
