"""The parameter record of the pinned checkpoints, written as a user runs the command."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SOURCES = os.environ.get("GFNET_ADAPTIVENN_SOURCES")
pytestmark = pytest.mark.skipif(
    not SOURCES, reason="set GFNET_ADAPTIVENN_SOURCES to a root written by `python -m gfnet_adaptivenn.sources`",
)

# Parameters run up to each glimpse, derived separately from the checkpoints' state dicts, module by module.
ADAPTIVENN = [21_993_448, 59_921_513, 67_411_281, 74_901_049, 82_390_817]
GFNET = {
    "gf_effb3_144_T4.pth.tar": [12_233_232, 26_651_938, 31_260_938, 37_405_938],
    "gf_mbv3l100_128_T3.pth.tar": [5_483_032, 12_803_378, 16_644_378],
}


def test_parameters_run_up_to_each_glimpse(tmp_path: Path) -> None:
    out = tmp_path / "active_params.json"
    subprocess.run((sys.executable, "-m", "gfnet_adaptivenn.active_params", "--root", str(SOURCES), "--out", str(out)),
                   check=True)
    record = json.loads(out.read_text())
    assert [budget["parameters"] for budget in record["adaptivenn_deit_s"]["per_k"].values()] == ADAPTIVENN
    assert {file: model["parameters_by_glimpse"] for file, model in record["gfnet"].items()} == GFNET
