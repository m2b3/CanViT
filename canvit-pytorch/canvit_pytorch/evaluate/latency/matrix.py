"""The paper's latency matrix (Appendix F): every model at every scene size, on CPU and on CUDA.

CPU runs float32 in eager mode at each thread count; CUDA runs each precision, compiled or not.
Each pass visits the configurations in a new random order, each in a fresh process. The machine
must be idle first: GPU utilization and processes, and the CPU load average, under thresholds.
"""

import logging
import os
import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from canvit_pytorch.evaluate.latency.measure import LatencyMeasurement, MeasurementProtocol, ModelName, Precision
from canvit_pytorch.evaluate.processes import run_in_fresh_process

log = logging.getLogger(__name__)


def physical_cores() -> int:
    """Distinct (socket, core) pairs in Linux's CPU topology; os.cpu_count() where that is unavailable."""
    cores = {
        ((cpu / "topology/physical_package_id").read_text(), (cpu / "topology/core_id").read_text())
        for cpu in Path("/sys/devices/system/cpu").glob("cpu[0-9]*")
        if (cpu / "topology/core_id").exists()
    }
    return len(cores) or os.cpu_count() or 1


@dataclass(frozen=True)
class IdleThresholds:
    max_gpu_utilization_pct: int = 5
    max_gpu_processes: int = 0
    max_load_average_1min: float = 2.0


def check_idle(*, check_gpu: bool, thresholds: IdleThresholds = IdleThresholds()) -> None:
    """Raise if the GPU (when check_gpu) or the CPU is busier than the thresholds allow."""
    if check_gpu:
        def query(*args: str) -> str:
            return subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"], check=True,
                                  capture_output=True, text=True).stdout
        utilization = max(int(line) for line in query("--query-gpu=utilization.gpu").split())
        processes = len(query("--query-compute-apps=pid").split())
        log.info("GPU utilization %d%%, %d compute processes", utilization, processes)
        if utilization > thresholds.max_gpu_utilization_pct or processes > thresholds.max_gpu_processes:
            raise RuntimeError(f"GPU busy: utilization {utilization}%, {processes} compute processes")
    load = os.getloadavg()[0]
    log.info("CPU load average over 1 min: %.2f", load)
    if load > thresholds.max_load_average_1min:
        raise RuntimeError(f"CPU busy: load average {load:.2f} > {thresholds.max_load_average_1min}")


@dataclass(frozen=True, kw_only=True)
class LatencyMatrix:
    """Latency and peak memory of every model at every scene size on CPU and CUDA, each in a fresh process."""

    models: tuple[ModelName, ...] = ("canvit", "dinov3-vitb16", "dinov3-vits16")
    cpu_scene_sizes_px: tuple[int, ...] = (128, 256, 512, 1024)
    cpu_threads: tuple[int, ...] = field(default_factory=lambda: tuple(sorted({1, physical_cores()})))
    """Thread counts for CPU runs; default: 1 and the number of physical cores."""
    cuda_scene_sizes_px: tuple[int, ...] = (128, 256, 512, 1024, 2048)
    cuda_dtypes: tuple[Precision, ...] = ("fp32", "amp-bf16")
    cuda_compiled: bool = True
    passes: int = 3
    """Each configuration is measured once per pass; the passes' iterations are pooled."""
    protocol: MeasurementProtocol = MeasurementProtocol()
    output_dir: Path = Path("results/latency")
    seed: int = 42
    """Seeds the order of configurations within each pass."""
    check_idle: bool = True
    dry_run: bool = False
    """Print the measurements instead of running them."""

    def measurements(self) -> list[LatencyMeasurement]:
        cpu = [
            LatencyMeasurement(
                model=model, device="cpu", scene_size_px=scene, dtype="fp32", compiled=False, num_threads=threads,
                protocol=self.protocol, output_dir=self.output_dir,
            )
            for model in self.models for scene in self.cpu_scene_sizes_px for threads in self.cpu_threads
        ]
        cuda = [
            LatencyMeasurement(
                model=model, device="cuda", scene_size_px=scene, dtype=dtype, compiled=self.cuda_compiled,
                protocol=self.protocol, output_dir=self.output_dir,
            )
            for model in self.models for scene in self.cuda_scene_sizes_px for dtype in self.cuda_dtypes
        ]
        measurements = []
        for pass_index in range(self.passes):
            order = random.Random(self.seed + pass_index)
            cpu_order, cuda_order = list(cpu), list(cuda)
            order.shuffle(cpu_order)
            order.shuffle(cuda_order)
            measurements += cpu_order + cuda_order
        return measurements

    def run(self) -> None:
        measurements = self.measurements()
        log.info("%d measurements over %d passes", len(measurements), self.passes)
        if self.dry_run:
            for measurement in measurements:
                print(measurement)
            return
        if self.check_idle:
            check_idle(check_gpu=any(m.device == "cuda" for m in measurements))
        failed = [str(m) for m in measurements if not run_in_fresh_process(m.run)]
        if failed:
            raise SystemExit(f"{len(failed)} of {len(measurements)} measurements failed: {failed}")
