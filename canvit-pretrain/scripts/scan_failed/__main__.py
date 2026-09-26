import tyro

from . import Config, main

main(tyro.cli(Config))
