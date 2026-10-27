"""Probe and policy runs as their trainers record them, staged for the Hub."""

import dataclasses
import json
from pathlib import Path

import pytest

from canvit_pytorch.hub.publish import Policy, Probe
from canvit_pytorch.hub.repos import FLAGSHIP
from canvit_pytorch.probes import SegmentationProbe
from canvit_pytorch.rl import ViewpointScorer
from canvit_pytorch.rl.candidates import CandidateViewpoints
from canvit_pytorch.rl.features import FEATURE_GROUPS
from canvit_pytorch.rl.observer import flagship_observer
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


def test_policy_run_is_staged_with_card_and_record(tmp_path: Path) -> None:
    run = tmp_path / "run"
    ViewpointScorer(
        observer=flagship_observer(canvas_grid_size=64), canvas_dim=8, feature_groups=FEATURE_GROUPS,
        candidates=CandidateViewpoints((0.5,), 2), width=8, blocks_per_stage=1, dueling=True,
    ).save_pretrained(run / "scorer")
    validations = [{"step": step, "training_loss": 0.1, "validation_cross_entropy": [0.9, value]}
                   for step, value in ((2, 0.75), (4, 0.5))]
    # A record written before the objectives took the paper's names; its card names the objective as the paper does.
    record = {"objective": "QRegression", "config": {"seed": 3, "num_glimpses": 2}, "selected_step": 4,
              "validations": validations}
    (run / "record.json").write_text(json.dumps(record))

    Policy(run_dir=run, repo="canvit/policy-test", release="9.9", out_dir=tmp_path / "staging").run()

    staged = tmp_path / "staging" / "policy-test"
    card = (staged / "README.md").read_text()
    assert all(text in card for text in ('"canvit/policy-test"', "canvit-pytorch>=9.9", "seed 3", "0.9000, 0.5000", "Q-learning"))
    assert json.loads((staged / "config.json").read_text())["training"] == record
    ViewpointScorer.from_pretrained(str(staged))
