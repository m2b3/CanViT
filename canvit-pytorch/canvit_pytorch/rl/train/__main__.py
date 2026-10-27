"""python -m canvit_pytorch.rl.train: train a learned viewing policy; print the selected scorer's directory."""

import logging

import tyro

from canvit_pytorch.rl.train import PolicyTraining


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    print(tyro.cli(PolicyTraining, config=(tyro.conf.CascadeSubcommandArgs,)).run())


if __name__ == "__main__":
    main()
