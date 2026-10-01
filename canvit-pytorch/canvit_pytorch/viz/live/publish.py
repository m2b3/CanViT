"""Publish a checked live export to the Hub as hub.repos.LIVE_MODEL: its graph, initial state and manifest, and a
card generated from the manifest and the parity report (hub.cards.live_model_card).

Refuses an export without a passing parity report on this very graph, or whose model or probe is no longer the
Hub's current revision. Stages hard links (the graph is hundreds of MB) and README.md in out_dir/<repo name>, and
uploads only with --push, to a new private repo (made public on the Hub once reviewed).
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from canvit_pytorch.hub import cards
from canvit_pytorch.hub.publish import upload
from canvit_pytorch.hub.repos import LIVE_MODEL
from canvit_pytorch.project import HUB_ORGANIZATION
from canvit_pytorch.viz.live.export import read_manifest, sha256_hex
from canvit_pytorch.viz.live.parity import COMPARED, MAX_REL_L2, MIN_ARGMAX_AGREEMENT
from canvit_pytorch.viz.released_model import hub_identity

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Publish:
    """Stage (and with --push, upload) a live export that export and parity produced."""

    model_dir: Path
    out_dir: Path
    push: bool = False

    def run(self) -> Path:
        manifest = read_manifest(self.model_dir)
        report_path = self.model_dir / "parity" / "report.json"
        assert report_path.is_file(), f"{report_path}: no parity report, run parity first"
        parity = json.loads(report_path.read_text())
        assert parity["graph_sha256"] == manifest["graph"]["sha256"], f"{report_path}: made on another graph, run parity again"
        worst = parity["worst"]
        assert all(worst[key] <= MAX_REL_L2 for key in COMPARED), f"{report_path}: ONNX differs from PyTorch: {worst}"
        assert worst["min_argmax_agreement"] >= MIN_ARGMAX_AGREEMENT, f"{report_path}: labels differ: {worst}"
        for part in ("model", "readout"):
            current = hub_identity(manifest[part]["repo"])
            assert current["revision"] == manifest[part]["revision"], (
                f"{self.model_dir}: {part} {current['repo']} exported at {manifest[part]['revision']}, the Hub is at "
                f"{current['revision']}: export again")

        name = LIVE_MODEL.rsplit("/", 1)[-1]
        repo = f"{HUB_ORGANIZATION}/{name}"
        staged = self.out_dir / name
        staged.mkdir(parents=True, exist_ok=False)
        for record in (manifest["graph"], manifest["initial_state"]):
            source = self.model_dir / record["path"]
            assert sha256_hex(source) == record["sha256"], f"{source}: its SHA-256 differs from the manifest's"
            os.link(source, staged / record["path"])
        (staged / "manifest.json").write_bytes((self.model_dir / "manifest.json").read_bytes())
        (staged / "README.md").write_text(cards.live_model_card(repo=repo, manifest=manifest, parity=parity))
        upload(staged, repo, push=self.push)
        return staged
