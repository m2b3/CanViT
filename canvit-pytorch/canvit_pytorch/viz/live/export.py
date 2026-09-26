"""Export the released CanViT-B with its ADE20K probe as one ONNX graph per glimpse, fp32, for <canvit-live>.

Writes the graph (weights embedded), the initial state and manifest.json (docs/viz.md, "Live model").
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxscript
import torch

from canvit_pytorch.evaluate.config import NUM_GLIMPSES
from canvit_pytorch.policies import POLICIES, EntropyGuidedC2F, make_policy
from canvit_pytorch.policies.random import MIN_SCALE
from canvit_pytorch.preprocess import IMAGENET_MEAN, IMAGENET_STD
from canvit_pytorch.provenance import provenance
from canvit_pytorch.viz.live.step import GlimpseStep, StepInputs, StepOutputs, initial_inputs
from canvit_pytorch.viz.released_model import (
    CANVAS_GRID_SIZE,
    GLIMPSE_SIZE_PX,
    SCENE_SIZE_PX,
    load_released_segmenter,
)

log = logging.getLogger(__name__)

SCHEMA = "canvit-live-model-2ef08388-92e1-4d7d-b5ac-2922601b5aa0"
OPSET = 18
"""ONNX Runtime Web 1.30.0's WebGPU execution provider implements GridSample for opsets 16 to 19 only."""
GRAPH_FILE = "glimpse_step.onnx"
INITIAL_STATE_FILE = "initial_state.bin"
EG_C2F = "entropy_coarse_to_fine"


def sha256_hex(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_hex(path)}


def export_graph(step: GlimpseStep, inputs: StepInputs, path: Path) -> StepOutputs:
    """Write the step as a self-contained ONNX file; return its PyTorch outputs on inputs."""
    assert not any(module.training for module in step.modules()), "export in eval mode: dropout and batch norm fixed"
    with torch.inference_mode():
        outputs = step(*inputs)
        program = torch.onnx.export(
            step, tuple(inputs), input_names=list(StepInputs._fields), output_names=list(StepOutputs._fields),
            opset_version=OPSET, dynamo=True, external_data=False,
        )
    assert program is not None
    program.save(str(path), external_data=False)
    onnx.checker.check_model(str(path))
    return outputs


def read_manifest(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    assert manifest["schema"] == SCHEMA, f"{path}: schema {manifest['schema']}, expected {SCHEMA}"
    return manifest


@dataclass(frozen=True)
class Export:
    """Export the glimpse step graph and its initial state for the browser."""

    out_dir: Path
    canvas_grid_size: int = CANVAS_GRID_SIZE
    scene_size_px: int = SCENE_SIZE_PX
    glimpse_size_px: int = GLIMPSE_SIZE_PX

    def run(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        device = torch.device("cpu")
        released = load_released_segmenter(
            scene_size_px=self.scene_size_px, canvas_grid_size=self.canvas_grid_size, device=device,
        )
        model = released.model
        step = GlimpseStep(model, glimpse_size_px=self.glimpse_size_px).eval()
        scene = torch.zeros(1, 3, self.scene_size_px, self.scene_size_px)
        inputs = initial_inputs(model, scene, canvas_grid_size=self.canvas_grid_size)

        graph_path = self.out_dir / GRAPH_FILE
        log.info("Exporting %s (opset %d)", graph_path, OPSET)
        outputs = export_graph(step, inputs, graph_path)

        canvit = model.canvit
        parts = {
            "init_canvas_registers": canvit.init_canvas_registers[0],
            "init_canvas_patch": canvit.init_canvas_patch[0, 0],
            "init_recurrent_cls": canvit.init_recurrent_cls[0, 0],
        }
        state_path = self.out_dir / INITIAL_STATE_FILE
        torch.cat([p.detach().flatten() for p in parts.values()]).numpy().astype("<f4").tofile(state_path)

        policy = make_policy(
            EG_C2F, batch_size=1, device=device, num_glimpses=NUM_GLIMPSES, canvas_grid_size=self.canvas_grid_size,
            canvas_logits=model.logits,
        )
        assert isinstance(policy, EntropyGuidedC2F)
        spec = POLICIES[EG_C2F]
        manifest = {
            "schema": SCHEMA,
            "graph": {
                **file_record(graph_path), "opset": OPSET,
                "inputs": {name: list(t.shape) for name, t in zip(StepInputs._fields, inputs, strict=True)},
                "outputs": {name: list(t.shape) for name, t in zip(StepOutputs._fields, outputs, strict=True)},
            },
            "initial_state": {
                **file_record(state_path), "dtype": "float32",
                "parts": [{"name": name, "shape": list(p.shape)} for name, p in parts.items()],
            },
            "scene_px": self.scene_size_px,
            "glimpse_px": self.glimpse_size_px,
            "canvas_grid": self.canvas_grid_size,
            "num_canvas_registers": canvit.config.num_canvas_registers,
            "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
            "min_scale": MIN_SCALE,
            "policy": {
                "name": EG_C2F, "paper_name": spec.paper_name, "description": spec.description,
                "num_glimpses": NUM_GLIMPSES, "levels": [[list(tile) for tile in level] for level in policy.levels],
            },
            "model": released.model_record,
            "readout": released.readout_record,
            "provenance": {
                **provenance(device), "precision": "float32", "onnx": onnx.__version__, "onnxscript": onnxscript.__version__,
            },
        }
        path = self.out_dir / "manifest.json"
        path.write_text(json.dumps(manifest, indent=1) + "\n")
        log.info("Wrote %s: graph %.1f MB", path, manifest["graph"]["bytes"] / 1e6)
        return path


def initial_state(manifest: dict[str, Any], model_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """(canvas [1, R + G*G, D], recurrent CLS [1, 1, backbone_dim]) from the initial-state file, as the page builds
    them: the registers, then the canvas patch broadcast over the grid (CanViT.init_state)."""
    spec = manifest["initial_state"]
    flat = np.fromfile(model_dir / spec["path"], dtype="<f4")
    sizes = [int(np.prod(part["shape"])) for part in spec["parts"]]
    assert flat.size == sum(sizes), (spec, flat.size)
    registers, patch, recurrent_cls = np.split(flat, np.cumsum(sizes)[:-1])
    num_registers, canvas_dim = spec["parts"][0]["shape"]
    patches = np.broadcast_to(patch, (manifest["canvas_grid"] ** 2, canvas_dim))
    canvas = np.concatenate([registers.reshape(num_registers, canvas_dim), patches])[None]
    return np.ascontiguousarray(canvas, dtype=np.float32), recurrent_cls.reshape(1, 1, -1).copy()
