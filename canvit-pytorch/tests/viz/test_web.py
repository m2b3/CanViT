import json

import numpy as np
import torch
from PIL import Image

from canvit_pytorch.viewpoint import Viewpoint
from canvit_pytorch.viz.record import GlimpseRecord, Rollout
from canvit_pytorch.viz.web import SCHEMA, entropy_fraction, glimpse_box, write_bundle

G, D, C, PX = 4, 8, 5, 16


def _viewpoint(row: float, col: float, scale: float) -> Viewpoint:
    return Viewpoint(centers=torch.tensor([[row, col]]), scales=torch.tensor([scale]))


def _rollout(capture_writes: bool) -> Rollout:
    rng = np.random.default_rng(0)
    tokens = lambda: rng.normal(size=(G * G, D)).astype(np.float32)  # noqa: E731
    glimpses = tuple(
        GlimpseRecord(
            t=t, viewpoint=_viewpoint(0.5, -0.5, 0.5),
            crop=rng.integers(0, 256, size=(PX, PX, 3), dtype=np.uint8),
            canvas=tokens(), logits=rng.normal(size=(C, G, G)).astype(np.float32),
            write_residuals=(tokens(), tokens()) if capture_writes else (),
        )
        for t in range(3)
    )
    annotation = rng.integers(0, C, size=(32, 32)).astype(np.int64)
    return Rollout(scene=rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8), initial_canvas=tokens(),
                   annotation=annotation, glimpses=glimpses, canvas_grid_size=G, glimpse_size_px=PX)


def test_glimpse_box_geometry():
    assert glimpse_box(_viewpoint(0.0, 0.0, 1.0)).__dict__ == {"top": 0.0, "left": 0.0, "size": 1.0}
    # Bottom-left quadrant: rows grow downward, columns rightward.
    assert glimpse_box(_viewpoint(0.5, -0.5, 0.5)).__dict__ == {"top": 0.5, "left": 0.0, "size": 0.5}


def test_entropy_fraction_bounds():
    uniform = np.zeros((C, 2, 2))
    peaked = np.zeros((C, 2, 2))
    peaked[0] = 50.0
    np.testing.assert_allclose(entropy_fraction(uniform), 1.0)
    assert entropy_fraction(peaked).max() < 1e-6


def test_bundle_is_self_describing_and_lossless(tmp_path):
    rollout = _rollout(capture_writes=True)
    path = write_bundle(
        rollout, tmp_path, title="t", scene={}, model={}, readout={}, policy={}, pca_protocol="fixed-limits", provenance={},
    )
    manifest = json.loads(path.read_text())
    assert manifest["schema"] == SCHEMA and len(manifest["glimpses"]) == 3
    first = manifest["glimpses"][0]
    assert set(first["layers"]) == {"crop", "canvas", "labels", "entropy", "change", "write0", "write1"}
    for rel in first["layers"].values():
        assert (tmp_path / rel).is_file()
    labels = np.asarray(Image.open(tmp_path / first["layers"]["labels"]))
    np.testing.assert_array_equal(labels, rollout.glimpses[0].logits.argmax(axis=0))
    entropy = np.asarray(Image.open(tmp_path / first["layers"]["entropy"]))
    assert entropy.dtype == np.uint16
    np.testing.assert_allclose(entropy / 65535, entropy_fraction(rollout.glimpses[0].logits), atol=1 / 65535)
    assert np.asarray(Image.open(tmp_path / first["layers"]["canvas"])).shape == (G, G, 3)
    assert np.asarray(Image.open(tmp_path / manifest["scene"]["truth"])).shape == (G, G)
    assert 0 <= first["pixel_accuracy"] <= 1
