"""Publish a checked export to the Hub, CanViT and the probe apart: hub.repos.LIVE_CANVIT (the glimpse step graph,
the initial state, its manifest) and hub.repos.LIVE_PROBE (the readout graph, its manifest), each with a card
generated from its manifest and the parity report (hub.cards).

Refuses an export without a passing parity report on these very graphs, or whose model or probe is no longer the
Hub's current revision. Stages hard links (the CanViT graph is hundreds of MB) and README.md in
out_dir/<repo name>, and uploads only with --push, to new private repos (made public on the Hub once reviewed).
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from canvit_pytorch.hub import cards
from canvit_pytorch.hub.publish import upload
from canvit_pytorch.hub.repos import LIVE_CANVIT, LIVE_PROBE
from canvit_pytorch.project import HUB_ORGANIZATION
from canvit_pytorch.viz.live.export import CANVIT_DIR, PROBE_DIR, read_manifests, sha256_hex
from canvit_pytorch.viz.live.parity import COMPARED, MAX_REL_L2, MIN_ARGMAX_AGREEMENT
from canvit_pytorch.viz.released_model import hub_identity

log = logging.getLogger(__name__)


def hub_name(repo: str) -> str:
    return f"{HUB_ORGANIZATION}/{repo.rsplit('/', 1)[-1]}"


def stage(source: Path, manifest: dict[str, Any], files: list[dict[str, Any]], card: str, staged: Path) -> None:
    """Hard links of the files (checked against the manifest), the manifest and the card, in staged/."""
    staged.mkdir(parents=True, exist_ok=False)
    for record in files:
        path = source / record["path"]
        assert sha256_hex(path) == record["sha256"], f"{path}: its SHA-256 differs from the manifest's"
        os.link(path, staged / record["path"])
    (staged / "manifest.json").write_bytes((source / "manifest.json").read_bytes())
    (staged / "README.md").write_text(card)
    assert json.loads((staged / "manifest.json").read_text()) == manifest


@dataclass(frozen=True)
class Publish:
    """Stage (and with --push, upload) the two repos of an export that export and parity produced."""

    model_dir: Path
    """The export: its canvit/ and probe/ directories and parity/report.json."""
    out_dir: Path
    push: bool = False

    def run(self) -> Path:
        canvit, probe = read_manifests(self.model_dir)
        report_path = self.model_dir / "parity" / "report.json"
        assert report_path.is_file(), f"{report_path}: no parity report, run parity first"
        parity = json.loads(report_path.read_text())
        graphs = {CANVIT_DIR: canvit["graph"]["sha256"], PROBE_DIR: probe["graph"]["sha256"]}
        assert parity["graphs_sha256"] == graphs, f"{report_path}: made on other graphs, run parity again"
        worst = parity["worst"]
        assert all(worst[key] <= MAX_REL_L2 for key in COMPARED), f"{report_path}: ONNX differs from PyTorch: {worst}"
        assert worst["min_argmax_agreement"] >= MIN_ARGMAX_AGREEMENT, f"{report_path}: labels differ: {worst}"
        for record in (canvit["model"], probe["readout"]):
            current = hub_identity(record["repo"])
            assert current["revision"] == record["revision"], (
                f"{record['repo']} exported at {record['revision']}, the Hub is at {current['revision']}: export again")

        canvit_repo, probe_repo = hub_name(LIVE_CANVIT), hub_name(LIVE_PROBE)
        staged = {repo: self.out_dir / repo.rsplit("/", 1)[-1] for repo in (canvit_repo, probe_repo)}
        stage(self.model_dir / CANVIT_DIR, canvit, [canvit["graph"], canvit["initial_state"]],
              cards.live_canvit_card(repo=canvit_repo, probe_repo=probe_repo, manifest=canvit, parity=parity),
              staged[canvit_repo])
        stage(self.model_dir / PROBE_DIR, probe, [probe["graph"]],
              cards.live_probe_card(repo=probe_repo, canvit_repo=canvit_repo, manifest=probe, parity=parity),
              staged[probe_repo])
        for repo, directory in staged.items():
            upload(directory, repo, push=self.push)
        return self.out_dir
