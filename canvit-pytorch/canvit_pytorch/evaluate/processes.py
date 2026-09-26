"""Running each job of a sweep in a fresh Python process: memory, compiled kernels and global state never carry over."""

import logging
import multiprocessing
from collections.abc import Callable


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _run(job: Callable[[], object]) -> None:
    configure_logging()
    job()


def run_in_fresh_process(job: Callable[[], object]) -> bool:
    """Call job() in a spawned process; whether it exited cleanly. job must be picklable, like a config's method."""
    process = multiprocessing.get_context("spawn").Process(target=_run, args=(job,))
    process.start()
    process.join()
    return process.exitcode == 0
