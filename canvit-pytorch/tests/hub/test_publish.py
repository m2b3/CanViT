"""Probe runs as specialize.ade20k records them, staged for the Hub under the probe-name grammar."""

import dataclasses
import json
from pathlib import Path

import pytest

from canvit_pytorch.hub.publish import Probe
from canvit_pytorch.hub.repos import FLAGSHIP
from canvit_pytorch.probes import SegmentationProbe
from canvit_pytorch.specialize.ade20k.canvas_probe import CanvasProbeConfig
from canvit_pytorch.specialize.ade20k.config import ProbeTrainingConfig
from canvit_pytorch.specialize.ade20k.dinov3_probe import DINOv3ProbeConfig
from canvit_pytorch.specialize.ade20k.record import RunDirectory


@pytest.mark.parametrize(("setup", "embed_dim", "use_ln", "expected_name"), [
    (CanvasProbeConfig(training=ProbeTrainingConfig(output_dir=Path("unused")), pretrained_repo=FLAGSHIP,
                       canvas_grid_size=64), 1024, True, "probe-ade20k-40k-s512-c64-in21k"),
    (DINOv3ProbeConfig(training=ProbeTrainingConfig(output_dir=Path("unused")), variant="vits16", input_size_px=128),
     384, False, "probe-ade20k-40k-dv3s-128px"),
])
def test_probe_run_is_staged_with_card_and_record(
    tmp_path: Path, setup: CanvasProbeConfig | DINOv3ProbeConfig, embed_dim: int, use_ln: bool, expected_name: str,
) -> None:
    run = RunDirectory(tmp_path / "run", config_type=type(setup).__name__, config=dataclasses.asdict(setup),
                       comet_experiment_key="0" * 32, provenance={"git_commit": None})
    probe = SegmentationProbe(embed_dim=embed_dim, num_classes=150, dropout=0.1, use_ln=use_ln)
    run.record_validation(step=40_000, validation_miou=[0.4], probe=probe)

    Probe(run_dir=run.path, out_dir=tmp_path / "staging").run()

    staged = tmp_path / "staging" / expected_name
    assert f'"canvit/{expected_name}"' in (staged / "README.md").read_text()
    assert json.loads((staged / "config.json").read_text())["training"]["validation_miou"] == [0.4]
    SegmentationProbe.from_pretrained(str(staged))
