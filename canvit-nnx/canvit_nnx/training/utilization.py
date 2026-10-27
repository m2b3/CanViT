"""Hardware utilization of a compiled training step, from XLA's own count of its operations.

XLA counts the operations of the compiled step per device, recomputation in
backward passes included, so FLOPs per second against the device's peak is the
hardware FLOPs utilization (HFU). Pallas kernels are opaque to that count: on
TPU it leaves out the fused Canvas Attention kernels, and HFU reads low by
their share. Bytes accessed counts every operand read and written, including
those served from on-chip memory, so its rate can exceed the HBM bandwidth.
"""

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import jax

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DevicePeak:
    """Per JAX device: a chip's published peak divided by the JAX devices it exposes."""

    bfloat16_flops_per_second: float
    hbm_bytes_per_second: float


DEVICE_PEAKS: dict[str, DevicePeak] = {
    # Published per-chip bfloat16 peaks and HBM bandwidths (cloud.google.com/tpu/docs/<chip>, read 2026-07-18),
    # divided by the JAX devices each chip exposes: one, except two on TPU7x.
    "TPU v5 lite": DevicePeak(197e12, 800 * 2**30),
    "TPU v5": DevicePeak(459e12, 2765e9),
    "TPU v6 lite": DevicePeak(918e12, 1638e9),
    "TPU7x": DevicePeak(2307e12 / 2, 7380e9 / 2),
}
"""Keyed by jax.Device.device_kind."""


class Compiled(Protocol):
    def cost_analysis(self) -> Any: ...


@dataclass(frozen=True)
class StepCost:
    flops: float  # per device and step
    bytes_accessed: float  # per device and step
    peak: DevicePeak | None  # None on hardware without a listed peak

    @staticmethod
    def of(compiled: Compiled) -> "StepCost":
        analysis = compiled.cost_analysis()
        assert isinstance(analysis, dict), f"no cost analysis on {jax.default_backend()}: {analysis}"
        kind = jax.devices()[0].device_kind
        peak = DEVICE_PEAKS.get(kind)
        if peak is None:
            log.info("No listed peak for %s devices: utilization is not reported", kind)
        return StepCost(flops=float(analysis["flops"]), bytes_accessed=float(analysis["bytes accessed"]), peak=peak)

    def utilization(self, seconds_per_step: float) -> dict[str, float]:
        """HFU and bytes accessed per second against HBM bandwidth; empty without a peak."""
        if self.peak is None:
            return {}
        return {
            "hfu": self.flops / seconds_per_step / self.peak.bfloat16_flops_per_second,
            "bytes_per_hbm_bandwidth": self.bytes_accessed / seconds_per_step / self.peak.hbm_bytes_per_second,
        }
