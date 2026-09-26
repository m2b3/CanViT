"""Record CanViT-B with an ADE20K probe on one image and write a web bundle.

    uv run --extra viz python -m canvit_pytorch.viz rollout --image IMAGE --out site/data/NAME ...
    uv run --extra viz python -m canvit_pytorch.viz path --image IMAGE --keypoints 0,0,1 0.2,-0.3,0.25 ...
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
from PIL import Image

from canvit_pytorch.benchmarks.ade20k import decode_annotation
from canvit_pytorch.model.segmentation import CanViTForSemanticSegmentation
from canvit_pytorch.policies import POLICIES, PolicyName, make_policy
from canvit_pytorch.preprocess import imagenet_denormalize, preprocess, preprocess_labels
from canvit_pytorch.provenance import provenance
from canvit_pytorch.viz.path import closed_bezier, sample_path
from canvit_pytorch.viz.path_bundle import record_path, write_path_bundle
from canvit_pytorch.viz.record import record
from canvit_pytorch.viz.released_model import (
    CANVAS_GRID_SIZE,
    GLIMPSE_SIZE_PX,
    SCENE_SIZE_PX,
    load_released_segmenter,
)
from canvit_pytorch.viz.web import PCAProtocol, write_bundle

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Scene:
    image: Path
    annotation: Path | None
    """The image's ADE20K annotation (a label PNG), or None for an image without one."""
    source: str
    """Where the image comes from; bundles are published, so it is required."""
    attribution: str
    license: str
    out: Path
    title: str
    canvas_grid_size: int = CANVAS_GRID_SIZE
    glimpse_size_px: int = GLIMPSE_SIZE_PX
    scene_size_px: int = SCENE_SIZE_PX
    device: str = "cpu"


def _load(scene: Scene) -> tuple[CanViTForSemanticSegmentation, torch.Tensor, torch.Tensor | None, dict[str, Any], dict[str, Any]]:
    """(model, [1, 3, S, S] image, [S, S] annotation, model and readout records for the manifest)."""
    device = torch.device(scene.device)
    released = load_released_segmenter(
        scene_size_px=scene.scene_size_px, canvas_grid_size=scene.canvas_grid_size, device=device,
    )
    image = preprocess(scene.scene_size_px)(Image.open(scene.image).convert("RGB"))
    assert isinstance(image, torch.Tensor)
    annotation = None
    if scene.annotation is not None:
        pixels = preprocess_labels(scene.scene_size_px)(Image.open(scene.annotation))
        assert isinstance(pixels, torch.Tensor)
        annotation = decode_annotation(pixels[0])
    return released.model, image.unsqueeze(0).to(device), annotation, released.model_record, released.readout_record


def _scene_record(scene: Scene) -> dict[str, Any]:
    record = {"source": scene.source, "attribution": scene.attribution, "license": scene.license}
    return record | ({"annotation_source": scene.annotation.name} if scene.annotation is not None else {})


@dataclass(frozen=True)
class Rollout:
    """One rollout under a viewing policy of the paper, glimpse by glimpse."""

    scene: Scene
    policy: PolicyName = "entropy_coarse_to_fine"
    num_glimpses: int = 21
    pca_protocol: PCAProtocol = "fixed-limits"
    capture_writes: bool = False
    seed: int = 0

    def run(self) -> Path:
        model, image, annotation, model_record, readout_record = _load(self.scene)
        torch.manual_seed(self.seed)
        policy = make_policy(
            self.policy, batch_size=1, device=image.device, num_glimpses=self.num_glimpses,
            canvas_grid_size=self.scene.canvas_grid_size, canvas_logits=model.logits,
        )
        rollout = record(
            model, image, policy, num_glimpses=self.num_glimpses, canvas_grid_size=self.scene.canvas_grid_size,
            glimpse_size_px=self.scene.glimpse_size_px, capture_writes=self.capture_writes, annotation=annotation,
        )
        spec = POLICIES[self.policy]
        return write_bundle(
            rollout, self.scene.out, title=self.scene.title, scene=_scene_record(self.scene),
            model=model_record, readout=readout_record,
            policy={"name": self.policy, "paper_name": spec.paper_name, "description": spec.description,
                    "deterministic": spec.deterministic, "seed": self.seed},
            pca_protocol=self.pca_protocol,
            provenance={**provenance(image.device), "precision": "float32"},
        )


@dataclass(frozen=True)
class SmoothPath:
    """A closed smooth path through hand-chosen viewpoints, one glimpse per sample, with the canvas carried and
    reset at every viewpoint."""

    scene: Scene
    keypoints: tuple[tuple[float, float, float], ...]
    """(row, col, scale) viewpoints the path passes through, in order; it returns to the first."""
    num_samples: int = 300
    duration_ms: int = 5000

    def run(self) -> Path:
        model, image, annotation, model_record, readout_record = _load(self.scene)
        segments = closed_bezier(list(self.keypoints))
        viewpoints = sample_path(segments, self.num_samples)
        annotation_array = None if annotation is None else annotation.numpy().astype(np.int64)
        conditions, crops = record_path(
            model, image, viewpoints, canvas_grid_size=self.scene.canvas_grid_size,
            glimpse_size_px=self.scene.glimpse_size_px, annotation=annotation_array,
        )
        scene_pixels = imagenet_denormalize(image[0].cpu()).permute(1, 2, 0).numpy()
        return write_path_bundle(
            self.scene.out, title=self.scene.title, scene=(scene_pixels * 255).round().astype(np.uint8),
            scene_meta=_scene_record(self.scene), annotation=annotation_array,
            keypoints=list(self.keypoints), segments=segments, viewpoints=viewpoints, duration_ms=self.duration_ms,
            conditions=conditions, crops=crops, canvas_grid_size=self.scene.canvas_grid_size,
            model=model_record, readout=readout_record,
            provenance={**provenance(image.device), "precision": "float32"},
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    command = tyro.extras.subcommand_cli_from_dict({"rollout": Rollout, "path": SmoothPath})
    log.info(f"Wrote {command.run()}")
