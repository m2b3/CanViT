"""Progress of long loops: a tqdm bar when stderr is a terminal, otherwise the same status as log lines."""

import io
import logging
import sys
from collections.abc import Iterable

from tqdm import tqdm

log = logging.getLogger(__name__)

LOG_INTERVAL_SECONDS = 60.0


class _LogLines(io.TextIOBase):
    """A stream for tqdm that logs each new status it writes; tqdm starts a status with a carriage return, and
    repeats the last one when the loop ends."""

    def __init__(self) -> None:
        super().__init__()
        self.last_status = ""

    def write(self, text: str) -> int:
        if (status := text.strip()) and status != self.last_status:
            log.info(status)
            self.last_status = status
        return len(text)


def progress[T](iterable: Iterable[T], *, description: str) -> Iterable[T]:
    """iterable, reporting how far the loop is through it: a tqdm bar when stderr is a terminal; otherwise a log line
    at the start, at most every LOG_INTERVAL_SECONDS, and at the end."""
    if sys.stderr.isatty():
        return tqdm(iterable, desc=description)
    # miniters=1: tqdm checks the time after every iteration. With its adaptive default, its monitor thread redraws
    # a status between iterations, from a count that lags the loop.
    return tqdm(iterable, desc=description, file=_LogLines(), mininterval=LOG_INTERVAL_SECONDS, miniters=1)
