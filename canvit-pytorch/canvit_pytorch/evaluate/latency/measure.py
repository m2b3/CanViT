"""Latency of one forward pass at batch size 1: CanViT processing one glimpse, or DINOv3 one image.

The protocol of the paper's Appendix F: untimed warmup iterations, then timed iterations, each
between two device synchronizations, at least min_iters of them and until time_budget_s or
max_iters; peak memory over the timed iterations on CUDA. The paper reports the minimum latency.

The record is a JSONL file bench_<run id>.jsonl, written as the measurement proceeds: a "meta" row
describing the configuration, then {"type": "warmup" | "iter", "i", "ms"} rows ("iter" rows also
carry "wall_s", the time since timing began), then on CUDA {"type": "peak_mem", "peak_mem_mb"}.
"""

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import torch

from canvit_pytorch.evaluate.config import GLIMPSE_SIZE_PX
from canvit_pytorch.hub.repos import RELEASED_SCENE_SIZE_PX
from canvit_pytorch.model import CanViT, CanViTConfig
from canvit_pytorch.model.backbone import BACKBONES
from canvit_pytorch.provenance import provenance
from canvit_pytorch.teacher import DINOV3_REPOS, DINOv3Variant, load_teacher
from canvit_pytorch.viewpoint import Viewpoint, sample_at_viewpoint

log = logging.getLogger(__name__)

ModelName = Literal["canvit", "dinov3-vitb16", "dinov3-vits16"]
Device = Literal["cuda", "cpu", "mps"]
Precision = Literal["fp32", "amp-bf16"]

DINOV3_VARIANTS: dict[ModelName, DINOv3Variant] = {"dinov3-vitb16": "vitb16", "dinov3-vits16": "vits16"}
BATCH_SIZE = 1
PATCH_SIZE_PX = BACKBONES["vitb16"].patch_size  # CanViT-B's, and DINOv3 ViT-S/16's and ViT-B/16's


@dataclass(frozen=True)
class MeasurementProtocol:
    """Untimed warmup iterations, then timed iterations until the time budget or max_iters, and at least min_iters."""

    warmup_iters: int = 3
    min_iters: int = 5
    """Timed iterations run even when they exceed the time budget."""
    max_iters: int = 500
    time_budget_s: float = 20.0

    def __post_init__(self) -> None:
        assert self.warmup_iters >= 0 and 1 <= self.min_iters <= self.max_iters and self.time_budget_s > 0, self


def synchronize(device: torch.device) -> None:
    match device.type:
        case "cuda":
            torch.cuda.synchronize(device)
        case "mps":
            torch.mps.synchronize()


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple Silicon (MPS)"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return "CPU"


@dataclass(frozen=True, kw_only=True)
class LatencyMeasurement:
    """Latency of one forward pass at batch size 1, and peak memory on CUDA, for one configuration."""

    model: ModelName = "canvit"
    device: Device = "cuda"
    scene_size_px: int = RELEASED_SCENE_SIZE_PX
    """DINOv3's input size. CanViT takes one 128 px glimpse onto a canvas with one patch per 16 px of the scene."""
    dtype: Precision = "amp-bf16"
    """Weights stay float32; amp-bf16 computes under bfloat16 autocast."""
    compiled: bool = False
    num_threads: int = 0
    """CPU threads; 0 keeps PyTorch's default."""
    protocol: MeasurementProtocol = MeasurementProtocol()
    output_dir: Path = Path("results/latency")

    def __post_init__(self) -> None:
        assert self.scene_size_px % PATCH_SIZE_PX == 0, f"{self.scene_size_px} px is not a whole number of patches"

    @property
    def canvas_grid_size(self) -> int:
        return self.scene_size_px // PATCH_SIZE_PX

    def run_id(self, timestamp: str) -> str:
        grid = f"_cg{self.canvas_grid_size}" if self.model == "canvit" else ""
        device = f"_{self.device}" if self.device != "cuda" else ""
        threads = f"_t{self.num_threads}" if self.num_threads > 0 else ""
        compiled = "c" if self.compiled else "e"
        return f"{self.model}_{compiled}_{self.dtype}_{self.scene_size_px}px{grid}{device}{threads}_{timestamp}"

    def forward_pass(self, device: torch.device) -> Callable[[], object]:
        """The timed call, with its model loaded and its input on device."""
        if self.model == "canvit":
            canvit = CanViT(CanViTConfig()).to(device).eval()
            if self.compiled:
                canvit.compile()
            glimpse_size_px = GLIMPSE_SIZE_PX
            viewpoint = Viewpoint.full_scene(batch_size=BATCH_SIZE, device=device)
            image = torch.randn(BATCH_SIZE, 3, glimpse_size_px, glimpse_size_px, device=device)
            glimpse = sample_at_viewpoint(spatial=image, viewpoint=viewpoint, glimpse_size_px=glimpse_size_px)
            log.info("CanViT-B: %.1fM parameters, %d px glimpse, %d² canvas",
                     sum(p.numel() for p in canvit.parameters()) / 1e6, glimpse_size_px, self.canvas_grid_size)

            def canvit_forward() -> object:
                state = canvit.init_state(batch_size=BATCH_SIZE, canvas_grid_size=self.canvas_grid_size)
                return canvit(glimpse=glimpse, state=state, viewpoint=viewpoint)

            return canvit_forward
        teacher = load_teacher(DINOV3_REPOS[DINOV3_VARIANTS[self.model]], device)
        if self.compiled:
            teacher.compile()
        images = torch.randn(BATCH_SIZE, 3, self.scene_size_px, self.scene_size_px, device=device)
        log.info("%s: %.1fM parameters, %d px input", self.model, sum(p.numel() for p in teacher.parameters()) / 1e6,
                 self.scene_size_px)
        return lambda: teacher(images)

    def meta(self, *, device: torch.device, run_id: str, timestamp: str) -> dict[str, object]:
        record: dict[str, object] = {
            "type": "meta", "run_id": run_id, "timestamp": timestamp,
            "model": self.model, "device": self.device, "scene_px": self.scene_size_px, "dtype": self.dtype,
            "compiled": self.compiled, "batch_size": BATCH_SIZE,
            "num_threads": self.num_threads, "num_threads_actual": torch.get_num_threads(),
            "warmup_iters": self.protocol.warmup_iters, "min_iters": self.protocol.min_iters,
            "max_iters": self.protocol.max_iters, "time_budget_s": self.protocol.time_budget_s,
            "device_name": device_name(device), "torch_version": torch.__version__,
            "provenance": provenance(device),
        }
        if device.type == "cuda":
            record["device_mem_gb"] = round(torch.cuda.get_device_properties(device).total_memory / 1e9, 1)
        if self.model == "canvit":
            record |= {"canvas_grid": self.canvas_grid_size, "glimpse_px": GLIMPSE_SIZE_PX}
        return record

    @torch.inference_mode()
    def run(self) -> Path:
        if self.num_threads > 0:
            torch.set_num_threads(self.num_threads)
        device = torch.device(self.device)
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        run_id = self.run_id(timestamp)
        output = self.output_dir / f"bench_{run_id}.jsonl"
        forward_pass = self.forward_pass(device)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as record, torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=self.dtype == "amp-bf16",
        ):
            def write(row: dict[str, object]) -> None:
                record.write(json.dumps(row) + "\n")
                record.flush()

            def timed() -> float:
                synchronize(device)
                start = time.perf_counter()
                forward_pass()
                synchronize(device)
                return (time.perf_counter() - start) * 1000

            write(self.meta(device=device, run_id=run_id, timestamp=timestamp))
            for i in range(self.protocol.warmup_iters):
                ms = timed()
                log.info("warmup %d: %.1f ms", i, ms)
                write({"type": "warmup", "i": i, "ms": round(ms, 4)})
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            protocol, i, timing_start = self.protocol, 0, time.perf_counter()
            while True:
                ms = timed()
                wall_s = time.perf_counter() - timing_start
                write({"type": "iter", "i": i, "ms": round(ms, 4), "wall_s": round(wall_s, 3)})
                if i <= 3 or i % 50 == 0:
                    log.info("iteration %d: %.2f ms (%.1f s)", i, ms, wall_s)
                i += 1
                if i >= protocol.min_iters and (wall_s >= protocol.time_budget_s or i >= protocol.max_iters):
                    break
            if device.type == "cuda":
                peak_mem_mb = round(torch.cuda.max_memory_allocated(device) / 1e6, 1)
                log.info("peak memory: %.1f MB", peak_mem_mb)
                write({"type": "peak_mem", "peak_mem_mb": peak_mem_mb})
        log.info("%d timed iterations -> %s", i, output)
        return output
