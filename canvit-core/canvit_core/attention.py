def canvas_attention_schedule(
    *,
    num_blocks: int,
    rw_stride: int,
    enable_reads: bool,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    assert min(num_blocks, rw_stride) > 0, (num_blocks, rw_stride)
    positions = tuple(range(rw_stride - 1, num_blocks, rw_stride))
    reads, writes = positions[0::2], positions[1::2]
    if not writes or writes[-1] != num_blocks - 1:
        writes += (num_blocks - 1,)
    return reads if enable_reads else (), writes
