def canvas_attention_schedule(
    *,
    num_blocks: int,
    rw_stride: int,
    enable_reads: bool,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(reads, writes): the backbone blocks after which a Canvas Attention Read or Write runs.

    Every rw_stride blocks, Reads and Writes alternate, a Read first; the last block is always followed by a
    Write, so the canvas holds what the whole backbone computed. Without reads, the Writes keep these positions.
    """
    assert min(num_blocks, rw_stride) > 0, (num_blocks, rw_stride)
    positions = tuple(range(rw_stride - 1, num_blocks, rw_stride))
    reads, writes = positions[0::2], positions[1::2]
    if not writes or writes[-1] != num_blocks - 1:
        writes += (num_blocks - 1,)
    return reads if enable_reads else (), writes
