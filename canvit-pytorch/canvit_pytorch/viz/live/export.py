"""Export the released CanViT-B and its ADE20K probe as two float32 ONNX graphs for <canvit-live>, apart.

Writes out_dir/canvit/ (the glimpse step graph with its weights embedded, the initial state, manifest.json) and
out_dir/probe/ (the readout graph, manifest.json), each self-describing (docs/viz.md, "Live model").
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
from torch import nn

from canvit_pytorch.evaluate.config import NUM_GLIMPSES
from canvit_pytorch.policies import POLICIES, EntropyGuidedC2F, make_policy
from canvit_pytorch.policies.random import MIN_SCALE
from canvit_pytorch.preprocess import IMAGENET_MEAN, IMAGENET_STD
from canvit_pytorch.provenance import provenance
from canvit_pytorch.viz.live.step import (
    GlimpseStep,
    ProbeReadout,
    ReadoutInputs,
    ReadoutOutputs,
    StepInputs,
    StepOutputs,
    initial_inputs,
)
from canvit_pytorch.viz.released_model import (
    CANVAS_GRID_SIZE,
    GLIMPSE_SIZE_PX,
    SCENE_SIZE_PX,
    load_released_segmenter,
)

log = logging.getLogger(__name__)

CANVIT_SCHEMA = "canvit-live-canvit-0a49b16e-2f86-41ae-8b08-f83d94fb1732"
PROBE_SCHEMA = "canvit-live-probe-23117c20-1e8e-4edb-a56d-21018d157b8d"
CANVIT_DIR = "canvit"
PROBE_DIR = "probe"
CANVIT_GRAPH = "canvit_step.onnx"
PROBE_GRAPH = "probe.onnx"
INITIAL_STATE_FILE = "initial_state.bin"
OPSET = 18
"""ONNX Runtime Web 1.30.0's WebGPU execution provider implements GridSample for opsets 16 to 19 only."""
EG_C2F = "entropy_coarse_to_fine"


def sha256_hex(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_hex(path)}


Names = tuple[str, ...]


def export_graph(
    module: nn.Module, inputs: tuple[torch.Tensor, ...], input_names: Names, output_names: Names, path: Path,
) -> tuple[torch.Tensor, ...]:
    """Write the module as a self-contained ONNX file; return its PyTorch outputs on inputs."""
    assert not any(m.training for m in module.modules()), "export in eval mode: dropout and batch norm fixed"
    with torch.inference_mode():
        outputs = module(*inputs)
        program = torch.onnx.export(
            module, tuple(inputs), input_names=list(input_names), output_names=list(output_names),
            opset_version=OPSET, dynamo=True, external_data=False,
        )
    assert program is not None
    program.save(str(path), external_data=False)
    onnx.checker.check_model(str(path))
    return tuple(outputs)


def graph_record(
    path: Path, inputs: tuple[torch.Tensor, ...], input_names: Names, outputs: tuple[torch.Tensor, ...], output_names: Names,
) -> dict[str, Any]:
    return {
        **file_record(path), "opset": OPSET,
        "inputs": {name: list(t.shape) for name, t in zip(input_names, inputs, strict=True)},
        "outputs": {name: list(t.shape) for name, t in zip(output_names, outputs, strict=True)},
    }


def read_manifest(directory: Path, schema: str) -> dict[str, Any]:
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    assert manifest["schema"] == schema, f"{path}: schema {manifest['schema']}, expected {schema}"
    return manifest


def read_manifests(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """(CanViT's, the probe's) manifests of an export, checked to agree on the canvas."""
    canvit = read_manifest(out_dir / CANVIT_DIR, CANVIT_SCHEMA)
    probe = read_manifest(out_dir / PROBE_DIR, PROBE_SCHEMA)
    assert probe["graph"]["inputs"]["canvas"] == canvit["graph"]["outputs"]["next_canvas"], (canvit["graph"], probe["graph"])
    assert (probe["canvas_grid"], probe["num_canvas_registers"]) == (canvit["canvas_grid"], canvit["num_canvas_registers"])
    return canvit, probe


@dataclass(frozen=True)
class Export:
    """Export the glimpse step and the probe's readout for the browser, each with its manifest."""

    out_dir: Path
    canvas_grid_size: int = CANVAS_GRID_SIZE
    scene_size_px: int = SCENE_SIZE_PX
    glimpse_size_px: int = GLIMPSE_SIZE_PX

    def run(self) -> Path:
        device = torch.device("cpu")
        released = load_released_segmenter(
            scene_size_px=self.scene_size_px, canvas_grid_size=self.canvas_grid_size, device=device,
        )
        model = released.model
        canvit = model.canvit
        common = {"canvas_grid": self.canvas_grid_size, "num_canvas_registers": canvit.config.num_canvas_registers}
        export_provenance = {
            **provenance(device), "precision": "float32", "onnx": onnx.__version__, "onnxscript": onnxscript.__version__,
        }

        canvit_dir = self.out_dir / CANVIT_DIR
        canvit_dir.mkdir(parents=True, exist_ok=True)
        step_inputs = initial_inputs(canvit, torch.zeros(1, 3, self.scene_size_px, self.scene_size_px),
                                     canvas_grid_size=self.canvas_grid_size)
        step_path = canvit_dir / CANVIT_GRAPH
        log.info("Exporting %s (opset %d)", step_path, OPSET)
        step_outputs = export_graph(GlimpseStep(canvit, glimpse_size_px=self.glimpse_size_px).eval(), step_inputs,
                                    StepInputs._fields, StepOutputs._fields, step_path)
        parts = {
            "init_canvas_registers": canvit.init_canvas_registers[0],
            "init_canvas_patch": canvit.init_canvas_patch[0, 0],
            "init_recurrent_cls": canvit.init_recurrent_cls[0, 0],
        }
        state_path = canvit_dir / INITIAL_STATE_FILE
        torch.cat([p.detach().flatten() for p in parts.values()]).numpy().astype("<f4").tofile(state_path)
        canvit_manifest = {
            "schema": CANVIT_SCHEMA,
            "graph": graph_record(step_path, step_inputs, StepInputs._fields, step_outputs, StepOutputs._fields),
            "initial_state": {
                **file_record(state_path), "dtype": "float32",
                "parts": [{"name": name, "shape": list(p.shape)} for name, p in parts.items()],
            },
            "scene_px": self.scene_size_px,
            "glimpse_px": self.glimpse_size_px,
            **common,
            "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
            "min_scale": MIN_SCALE,
            "model": released.model_record,
            "provenance": export_provenance,
        }
        (canvit_dir / "manifest.json").write_text(json.dumps(canvit_manifest, indent=1) + "\n")

        probe_dir = self.out_dir / PROBE_DIR
        probe_dir.mkdir(parents=True, exist_ok=True)
        readout_inputs = ReadoutInputs(step_outputs[0])
        probe_path = probe_dir / PROBE_GRAPH
        log.info("Exporting %s (opset %d)", probe_path, OPSET)
        readout_outputs = export_graph(ProbeReadout(model).eval(), readout_inputs, ReadoutInputs._fields,
                                       ReadoutOutputs._fields, probe_path)
        policy = make_policy(
            EG_C2F, batch_size=1, device=device, num_glimpses=NUM_GLIMPSES, canvas_grid_size=self.canvas_grid_size,
            canvas_logits=model.logits,
        )
        assert isinstance(policy, EntropyGuidedC2F)
        spec = POLICIES[EG_C2F]
        probe_manifest = {
            "schema": PROBE_SCHEMA,
            "graph": graph_record(probe_path, readout_inputs, ReadoutInputs._fields, readout_outputs, ReadoutOutputs._fields),
            **common,
            "readout": released.readout_record,
            # EG-C2F reads this probe's entropy, so its tiles come with the probe.
            "policy": {
                "name": EG_C2F, "paper_name": spec.paper_name, "description": spec.description,
                "num_glimpses": NUM_GLIMPSES, "levels": [[list(tile) for tile in level] for level in policy.levels],
            },
            "provenance": export_provenance,
        }
        (probe_dir / "manifest.json").write_text(json.dumps(probe_manifest, indent=1) + "\n")
        read_manifests(self.out_dir)
        log.info("Wrote %s: CanViT graph %.1f MB, probe graph %.2f MB", self.out_dir,
                 canvit_manifest["graph"]["bytes"] / 1e6, probe_manifest["graph"]["bytes"] / 1e6)
        return self.out_dir


def initial_state(manifest: dict[str, Any], canvit_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """(canvas [1, R + G*G, D], recurrent CLS [1, 1, backbone_dim]) from CanViT's initial-state file, as the page
    builds them: the registers, then the canvas patch broadcast over the grid (CanViT.init_state)."""
    spec = manifest["initial_state"]
    flat = np.fromfile(canvit_dir / spec["path"], dtype="<f4")
    sizes = [int(np.prod(part["shape"])) for part in spec["parts"]]
    assert flat.size == sum(sizes), (spec, flat.size)
    registers, patch, recurrent_cls = np.split(flat, np.cumsum(sizes)[:-1])
    num_registers, canvas_dim = spec["parts"][0]["shape"]
    patches = np.broadcast_to(patch, (manifest["canvas_grid"] ** 2, canvas_dim))
    canvas = np.concatenate([registers.reshape(num_registers, canvas_dim), patches])[None]
    return np.ascontiguousarray(canvas, dtype=np.float32), recurrent_cls.reshape(1, 1, -1).copy()
