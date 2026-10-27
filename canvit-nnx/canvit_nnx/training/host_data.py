"""Host-side data loading: decoding on a thread pool, and a host's share of each global batch.

Pillow releases the interpreter lock while it decodes and resizes, so threads
decode in parallel. Every host walks the same items in the same order and
keeps the rows of each global batch that DataParallel.host_rows assigns it.
"""

import itertools
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor


def map_in_order[T, U](function: Callable[[T], U], items: Iterator[T], *, threads: int) -> Iterator[U]:
    """function over items on a thread pool, in order, at most 2 × threads ahead of the consumer."""
    with ThreadPoolExecutor(threads) as pool:
        pending = deque(pool.submit(function, item) for item in itertools.islice(items, 2 * threads))
        for item in items:
            yield pending.popleft().result()
            pending.append(pool.submit(function, item))
        while pending:
            yield pending.popleft().result()


def host_items[T](items: Iterable[T], *, batch_size: int, rows: slice) -> Iterator[T]:
    """The items at this host's rows of consecutive global batches."""
    return (item for position, item in enumerate(items) if rows.start <= position % batch_size < rows.stop)


def host_batches[T, U](
    items: Iterable[T], decode: Callable[[T], U], *, num_items: int, batch_size: int, rows: slice, threads: int
) -> Iterator[list[U | None]]:
    """This host's decoded rows of each global batch of the num_items items; None fills the last global batch."""
    decoded = map_in_order(decode, host_items(items, batch_size=batch_size, rows=rows), threads=threads)
    for batch_start in range(0, num_items, batch_size):
        num_real = max(0, min(rows.stop, num_items - batch_start) - rows.start)
        batch: list[U | None] = [value for _, value in zip(range(num_real), decoded, strict=False)]
        if len(batch) < num_real:
            raise ValueError(f"the items end before {num_items}")
        yield batch + [None] * (rows.stop - rows.start - num_real)
    if next(decoded, None) is not None:
        raise ValueError(f"there are more than {num_items} items")
