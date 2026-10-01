"""An exported live model against PyTorch: ONNX Runtime on the CPU here, and the reference the browser check reads.

The reference is PyTorch (CPU, fp32) on one image: an EG-C2F episode, then glimpses at random viewpoints
drawn as in pretraining, the canvas carried throughout. ONNX Runtime replays the same viewpoints from the
exported initial state. Writes, under <model_dir>/parity/: report.json; scene.png, the scene exactly as
the reference saw it before normalization; episode.json, the viewpoints; and reference/*.bin, float32
outputs per glimpse, which <canvit-live> is compared with.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from numpy.typing import NDArray
from PIL import Image

from canvit_pytorch.policies import make_policy
from canvit_pytorch.policies.entropy import predictive_entropy
from canvit_pytorch.policies.random import random_viewpoints
from canvit_pytorch.preprocess import imagenet_denormalize, imagenet_normalize, preprocess
from canvit_pytorch.viewpoint import Viewpoint, sample_at_viewpoint
from canvit_pytorch.viz.live.export import EG_C2F, initial_state, read_manifest
from canvit_pytorch.viz.live.step import StepInputs, StepOutputs
from canvit_pytorch.viz.released_model import load_released_segmenter

log = logging.getLogger(__name__)

MAX_REL_L2 = 1e-3
"""Largest accepted ‖ONNX − PyTorch‖ / ‖PyTorch‖ per output. Rounding differences between correct fp32
implementations stay orders of magnitude below it; a wrong graph, input or state does not."""
MIN_ARGMAX_AGREEMENT = 0.999
"""Smallest accepted fraction of canvas cells with the same class; cells whose top two logits nearly tie may flip."""
COMPARED = ("canvas", "recurrent_cls", "logits", "entropy", "glimpse")


@dataclass(frozen=True)
class GlimpseOutputs:
    canvas: NDArray[np.float32]  # [1, R + G*G, canvas_dim]
    recurrent_cls: NDArray[np.float32]
    logits: NDArray[np.float32]  # [1, C, G, G]
    entropy: NDArray[np.float32]  # [1, G, G]
    glimpse: NDArray[np.float32]  # [1, 3, g, g]
    seconds: float


def float32_array(value: object) -> NDArray[np.float32]:
    assert isinstance(value, np.ndarray) and value.dtype == np.float32, type(value)
    return value


def errors(actual: NDArray[np.float32], reference: NDArray[np.float32]) -> dict[str, float]:
    diff = actual.astype(np.float64) - reference.astype(np.float64)
    return {
        "max_abs": float(np.abs(diff).max()),
        "max_abs_over_max_ref": float(np.abs(diff).max() / np.abs(reference).max()),
        "rel_l2": float(np.linalg.norm(diff) / np.linalg.norm(reference)),
    }


def argmax_agreement(logits: NDArray[np.float32], reference: NDArray[np.float32]) -> float:
    return float((logits.argmax(axis=1) == reference.argmax(axis=1)).mean())


def as_tuple(viewpoint: Viewpoint) -> tuple[float, float, float]:
    row, col = viewpoint.centers[0].tolist()
    return row, col, float(viewpoint.scales[0])


@dataclass(frozen=True)
class Parity:
    """Compare an exported live model with PyTorch on one image and write the browser check's reference."""

    model_dir: Path
    image: Path
    """A photograph, e.g. an ADE20K validation image."""
    num_random_glimpses: int = 8
    """Glimpses at random viewpoints after the EG-C2F episode."""
    seed: int = 0

    def run(self) -> Path:
        manifest = read_manifest(self.model_dir)
        device = torch.device("cpu")
        released = load_released_segmenter(
            scene_size_px=manifest["scene_px"], canvas_grid_size=manifest["canvas_grid"], device=device,
        )
        assert (released.model_record, released.readout_record) == (manifest["model"], manifest["readout"]), (
            f"{self.model_dir} was exported from other checkpoints than those loaded now"
        )
        model = released.model
        grid, glimpse_px = manifest["canvas_grid"], manifest["glimpse_px"]
        out_dir = self.model_dir / "parity"
        (out_dir / "reference").mkdir(parents=True, exist_ok=True)

        scene = preprocess(manifest["scene_px"])(Image.open(self.image).convert("RGB"))
        assert isinstance(scene, torch.Tensor)
        scene = scene.unsqueeze(0).contiguous()
        # The page normalizes 8-bit pixels itself; the scene's 8-bit form reproduces this tensor.
        pixels = (imagenet_denormalize(scene[0]).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        renormalized = imagenet_normalize(torch.from_numpy(pixels).permute(2, 0, 1).float() / 255)
        assert torch.allclose(renormalized, scene[0], atol=1e-5), "scene.png would not reproduce the scene"
        Image.fromarray(pixels).save(out_dir / "scene.png")

        torch.manual_seed(self.seed)
        num_egc2f = manifest["policy"]["num_glimpses"]
        policy = make_policy(
            EG_C2F, batch_size=1, device=device, num_glimpses=num_egc2f, canvas_grid_size=grid,
            canvas_logits=model.logits,
        )
        random = [random_viewpoints(batch_size=1, device=device) for _ in range(self.num_random_glimpses)]
        state = model.init_state(batch_size=1, canvas_grid_size=grid)
        viewpoints: list[Viewpoint] = []
        reference: list[GlimpseOutputs] = []
        with torch.inference_mode():
            for t in range(num_egc2f + self.num_random_glimpses):
                viewpoint = policy.step(t, state) if t < num_egc2f else random[t - num_egc2f]
                start = time.perf_counter()
                glimpse = sample_at_viewpoint(spatial=scene, viewpoint=viewpoint, glimpse_size_px=glimpse_px)
                logits, state = model(glimpse=glimpse, state=state, viewpoint=viewpoint)
                entropy = predictive_entropy(logits)
                seconds = time.perf_counter() - start
                viewpoints.append(viewpoint)
                reference.append(GlimpseOutputs(
                    state.canvas.numpy().copy(), state.recurrent_cls.numpy().copy(), logits.numpy(), entropy.numpy(),
                    glimpse.numpy(), seconds,
                ))

        session = ort.InferenceSession(
            str(self.model_dir / manifest["graph"]["path"]), providers=["CPUExecutionProvider"],
        )
        canvas, recurrent_cls = initial_state(manifest, self.model_dir)
        initial = model.init_state(batch_size=1, canvas_grid_size=grid).detach()
        assert np.array_equal(canvas, initial.canvas.numpy()) and np.array_equal(recurrent_cls, initial.recurrent_cls.numpy())
        replayed: list[GlimpseOutputs] = []
        for viewpoint in viewpoints:
            inputs = (scene.numpy(), canvas, recurrent_cls, viewpoint.centers.numpy(), viewpoint.scales.numpy())
            start = time.perf_counter()
            results = session.run(list(StepOutputs._fields), dict(zip(StepInputs._fields, inputs, strict=True)))
            seconds = time.perf_counter() - start
            canvas, recurrent_cls, logits, entropy, glimpse = (float32_array(r) for r in results)
            replayed.append(GlimpseOutputs(canvas, recurrent_cls, logits, entropy, glimpse, seconds))

        per_glimpse = [
            {"t": t, "viewpoint": as_tuple(viewpoints[t]), "policy": "EG-C2F" if t < num_egc2f else "random",
             **{key: errors(getattr(o, key), getattr(r, key)) for key in COMPARED},
             "argmax_agreement": argmax_agreement(o.logits, r.logits)}
            for t, (o, r) in enumerate(zip(replayed, reference, strict=True))
        ]
        report: dict[str, Any] = {
            "graph_sha256": manifest["graph"]["sha256"],
            "image": str(self.image),
            "comparison": "onnxruntime CPU execution provider against PyTorch CPU, float32, the same viewpoints",
            "onnxruntime": ort.__version__,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "egc2f_glimpses": num_egc2f,
            "worst": {key: max(g[key]["rel_l2"] for g in per_glimpse) for key in COMPARED}
            | {"max_abs_logits": max(g["logits"]["max_abs"] for g in per_glimpse),
               "max_abs_canvas": max(g["canvas"]["max_abs"] for g in per_glimpse),
               "min_argmax_agreement": min(g["argmax_agreement"] for g in per_glimpse)},
            # The first glimpse includes one-time setup in both runtimes.
            "warm_step_ms_median": {
                "torch_cpu": 1e3 * float(np.median([g.seconds for g in reference[1:]])),
                "onnxruntime_cpu": 1e3 * float(np.median([g.seconds for g in replayed[1:]])),
            },
            "per_glimpse": per_glimpse,
        }
        (out_dir / "report.json").write_text(json.dumps(report, indent=1) + "\n")
        self._write_reference(out_dir, viewpoints, reference, num_egc2f=num_egc2f)
        log.info("Worst over glimpses: %s; warm median step: %s", report["worst"], report["warm_step_ms_median"])

        worst = report["worst"]
        assert all(worst[key] <= MAX_REL_L2 for key in COMPARED), f"ONNX differs from PyTorch: {worst}"
        assert worst["min_argmax_agreement"] >= MIN_ARGMAX_AGREEMENT, f"ONNX labels differ from PyTorch: {worst}"
        return out_dir / "report.json"

    def _write_reference(
        self, out_dir: Path, viewpoints: list[Viewpoint], reference: list[GlimpseOutputs], *, num_egc2f: int,
    ) -> None:
        """Logits and entropy of every glimpse; the canvas and the glimpse where they change most (the first two
        glimpses, the end of the EG-C2F episode, the last glimpse)."""
        last = len(reference) - 1
        full_steps = sorted({0, 1, num_egc2f - 1, last})
        for t, outputs in enumerate(reference):
            names = ("logits", "entropy") + (("canvas", "glimpse") if t in full_steps else ())
            for name in names:
                getattr(outputs, name).astype("<f4").tofile(out_dir / "reference" / f"{name}_t{t:02d}.bin")
        episode = {
            "image": str(self.image), "scene": "scene.png", "egc2f_glimpses": num_egc2f,
            "viewpoints": [as_tuple(v) for v in viewpoints], "full_steps": full_steps,
            "reference": "reference/{name}_t{t:02d}.bin, little-endian float32, PyTorch CPU",
        }
        (out_dir / "episode.json").write_text(json.dumps(episode, indent=1) + "\n")
