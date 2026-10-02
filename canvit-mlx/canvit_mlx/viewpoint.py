from typing import NamedTuple

import mlx.core as mx


class Viewpoint(NamedTuple):
    centers: mx.array  # [B, 2], float32 (row, col) in scene coordinates
    scales: mx.array  # [B], float32 crop half-side

    @staticmethod
    def full_scene(*, batch_size: int) -> "Viewpoint":
        return Viewpoint(centers=mx.zeros((batch_size, 2)), scales=mx.ones((batch_size,)))


def grid_coords(*, size: int) -> mx.array:
    assert size > 0, size
    centers = (mx.arange(size, dtype=mx.float32) + 0.5) / size * 2 - 1
    return mx.stack(list(mx.meshgrid(centers, centers, indexing="ij")), axis=-1)


def viewpoint_grid_coords(viewpoint: Viewpoint, *, size: int) -> mx.array:
    assert viewpoint.centers.dtype == viewpoint.scales.dtype == mx.float32
    assert viewpoint.scales.ndim == 1 and viewpoint.centers.shape == (viewpoint.scales.shape[0], 2)
    return viewpoint.centers[:, None, None, :] + viewpoint.scales[:, None, None, None] * grid_coords(size=size)


def sample_at_viewpoint(*, spatial: mx.array, viewpoint: Viewpoint, glimpse_size_px: int) -> mx.array:
    """Bilinear, zero-padded crops of [B, H, W, C], matching align_corners=False."""
    batch_size, height, width, _ = spatial.shape
    assert batch_size == viewpoint.scales.shape[0], (spatial.shape, viewpoint.scales.shape)
    positions = viewpoint_grid_coords(viewpoint, size=glimpse_size_px)
    pixels = (positions + 1) * mx.array([height, width], dtype=mx.float32) / 2 - 0.5
    lower = mx.floor(pixels).astype(mx.int32)
    fraction = pixels - lower.astype(mx.float32)
    row, col = lower[..., 0], lower[..., 1]
    row_weight, col_weight = fraction[..., 0, None], fraction[..., 1, None]
    dtype = spatial.dtype
    spatial = spatial.astype(mx.float32)

    def gather(row: mx.array, col: mx.array) -> mx.array:
        valid = (row >= 0) & (row < height) & (col >= 0) & (col < width)
        values = spatial[mx.arange(batch_size)[:, None, None], mx.clip(row, 0, height - 1), mx.clip(col, 0, width - 1)]
        return mx.where(valid[..., None], values, 0.0)

    return (
        gather(row, col) * (1 - row_weight) * (1 - col_weight)
        + gather(row, col + 1) * (1 - row_weight) * col_weight
        + gather(row + 1, col) * row_weight * (1 - col_weight)
        + gather(row + 1, col + 1) * row_weight * col_weight
    ).astype(dtype)
