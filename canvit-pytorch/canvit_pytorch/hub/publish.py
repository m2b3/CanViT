"""Publish a trained checkpoint, probe or learned policy to the Hub: weights, config.json with the training record,
and its card.

Each command stages the repo in out_dir/<repo name>, loads the staged files
with the class users will load them with, and uploads only with --push, to a
new private repo (made public on the Hub once reviewed).

    python -m canvit_pytorch.hub.publish pretrained --checkpoint RUN/step-2000000.pt --repo NAME --out-dir staging
    python -m canvit_pytorch.hub.publish probe --run-dir PROBE_RUN --out-dir staging
    python -m canvit_pytorch.hub.publish policy --run-dir POLICY_RUN --repo NAME --release VERSION --out-dir staging
"""

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import tyro
from huggingface_hub import HfApi

from canvit_pytorch.hub import cards
from canvit_pytorch.hub.repos import (
    DINOV3_PROBE_MODEL_NAMES,
    PRETRAINED,
    PretrainingDataset,
    ade20k_probe_name,
    dinov3_ade20k_probe_name,
    hub_id,
)
from canvit_pytorch.model.pretraining import CanViTForPretraining
from canvit_pytorch.policies import POLICIES, PolicyName
from canvit_pytorch.pretrain import checkpoint
from canvit_pytorch.pretrain.ablations import ABLATIONS, AblationSlug
from canvit_pytorch.probes import SegmentationProbe
from canvit_pytorch.rl import ViewpointScorer
from canvit_pytorch.teacher import DINOV3_NAMES, DINOV3_PATCH_SIZE, DINOV3_REPOS, DINOv3Variant

log = logging.getLogger(__name__)


def _add_training_record(staged: Path, record: dict[str, Any]) -> None:
    config_path = staged / "config.json"
    config = json.loads(config_path.read_text())
    assert "training" not in config, config_path
    config_path.write_text(json.dumps(config | {"training": record}, indent=2, sort_keys=True, default=str) + "\n")


def upload(staged: Path, repo: str, *, push: bool) -> None:
    """Create `repo` private on the Hub and upload the staged directory, only with push."""
    if not push:
        log.info(f"Staged {repo} in {staged}; not pushed")
        return
    api = HfApi()
    api.create_repo(repo, private=True, exist_ok=False)
    api.upload_folder(repo_id=repo, folder_path=staged, commit_message="Initial release")
    log.info(f"Pushed https://huggingface.co/{repo} (private)")


@dataclass(frozen=True)
class Pretrained:
    """A pretraining checkpoint (canvit_pytorch.pretrain) as a CanViTForPretraining repo."""

    checkpoint: Path
    repo: str
    """The full Hub repo id."""
    out_dir: Path
    ablation: AblationSlug | None = None
    """The paper ablation the run trained, which the card names."""
    push: bool = False

    def run(self) -> None:
        saved = checkpoint.load(self.checkpoint, device=torch.device("cpu"))
        model = checkpoint.model_from_checkpoint(saved)
        staged = self.out_dir / Path(self.repo).name
        shutil.rmtree(staged, ignore_errors=True)
        model.save_pretrained(staged)
        last_job = saved["jobs"][-1]
        _add_training_record(staged, {
            "dataset": saved["dataset"], "teacher_repo": saved["teacher_repo"],
            "scene_size_px": saved["scene_size_px"], "glimpse_size_px": saved["glimpse_size_px"],
            "step": saved["step"], "comet_experiment_key": saved["comet_experiment_key"],
            "git_commit": last_job["provenance"]["git_commit"], "config": last_job["config"],
        })
        assert saved["dataset"] in PRETRAINED, saved["dataset"]
        facts = cards.PretrainingFacts(
            dataset=cast(PretrainingDataset, saved["dataset"]), teacher_repo=saved["teacher_repo"],
            scene_size_px=saved["scene_size_px"], glimpse_size_px=saved["glimpse_size_px"],
            canvas_grid_size=saved["teacher_patch_grid"], steps=saved["step"],
        )
        ablation = ABLATIONS[self.ablation] if self.ablation is not None else None
        (staged / "README.md").write_text(cards.pretrained_card(repo=self.repo, facts=facts, ablation=ablation))
        reloaded = CanViTForPretraining.from_pretrained(str(staged))
        for name, value in reloaded.state_dict().items():
            assert torch.equal(value, model.state_dict()[name]), name
        upload(staged, self.repo, push=self.push)


def _released_checkpoint(pretrained_repo: str) -> tuple[str, str]:
    """(Hub id, name in probe names) of a released checkpoint, given its Hub id or a local mirror directory."""
    names = {Path(repo).name: dataset for dataset, repo in PRETRAINED.items()}
    names |= {Path(a.released_repo).name: a.probe_model_name for a in ABLATIONS.values()}
    name = Path(pretrained_repo).name
    assert name in names, f"{pretrained_repo} is not a released checkpoint; its probes have no name yet"
    return hub_id(name), names[name]


def probe_card(record: dict[str, Any], *, embed_dim: int) -> tuple[str, str]:
    """(repo name, card) of a probe from its training record (canvit_pytorch.specialize.ade20k), the probe-name
    grammar of canvit_pytorch.hub.repos giving the name."""
    setup, training = record["config"], record["config"]["training"]
    low, high = training["scale_jitter_range"]
    probe_training = cards.ProbeTraining(
        steps=record["steps_trained"], selected_step=record["selected_step"], batch_size=training["batch_size"],
        peak_lr=training["peak_lr"], weight_decay=training["weight_decay"], warmup_steps=training["warmup_steps"],
        dropout=training["dropout"], crop_size_px=training["scene_size_px"], scale_jitter_range=(low, high),
        flip_probability=training["horizontal_flip_probability"],
        bfloat16=training["feature_dtype"] == "bfloat16",
    )
    match record["config_type"]:
        case "CanvasProbeConfig":
            pretrained_repo, model_name = _released_checkpoint(setup["pretrained_repo"])
            name = ade20k_probe_name(
                model_name, scene_size_px=training["scene_size_px"],
                canvas_grid_size=setup["canvas_grid_size"], steps=record["steps_trained"],
            )
            return name, cards.canvas_probe_card(repo=hub_id(name), facts=cards.CanvasProbeFacts(
                pretrained_repo=pretrained_repo, scene_size_px=training["scene_size_px"],
                canvas_grid_size=setup["canvas_grid_size"], canvas_dim=embed_dim,
                glimpse_size_px=setup["glimpse_size_px"], training_glimpses=setup["num_glimpses"],
                training_policy=POLICIES[cast(PolicyName, setup["training_policy"])].paper_name,
                training=probe_training,
            ))
        case "DINOv3ProbeConfig":
            variant = cast(DINOv3Variant, setup["variant"])
            name = dinov3_ade20k_probe_name(
                DINOV3_PROBE_MODEL_NAMES[variant], input_size_px=setup["input_size_px"], steps=record["steps_trained"],
            )
            return name, cards.dinov3_probe_card(repo=hub_id(name), facts=cards.DINOv3ProbeFacts(
                dinov3_repo=DINOV3_REPOS[variant], dinov3_name=DINOV3_NAMES[variant],
                input_size_px=setup["input_size_px"], patch_size=DINOV3_PATCH_SIZE, embed_dim=embed_dim,
                training=probe_training,
            ))
        case other:
            raise ValueError(f"unknown probe configuration {other!r}")


@dataclass(frozen=True)
class Probe:
    """A probe-training run directory (canvit_pytorch.specialize.ade20k) as a SegmentationProbe repo, named by
    the probe-name grammar of canvit_pytorch.hub.repos."""

    run_dir: Path
    out_dir: Path
    push: bool = False

    def run(self) -> None:
        record = json.loads((self.run_dir / "record.json").read_text())
        probe = SegmentationProbe.from_pretrained(str(self.run_dir / record["probe_dir"]))
        name, card = probe_card(record, embed_dim=probe.embed_dim)
        staged = self.out_dir / name
        shutil.rmtree(staged, ignore_errors=True)
        probe.save_pretrained(staged)
        _add_training_record(staged, {k: record[k] for k in (
            "selected_step", "validation_miou", "steps_trained", "config_type", "config", "comet_experiment_key",
            "provenance",
        )})
        (staged / "README.md").write_text(card)
        reloaded = SegmentationProbe.from_pretrained(str(staged))
        for key, value in reloaded.state_dict().items():
            assert torch.equal(value, probe.state_dict()[key]), key
        upload(staged, hub_id(name), push=self.push)


@dataclass(frozen=True)
class Policy:
    """A policy-training run directory (canvit_pytorch.rl.train) as a ViewpointScorer repo."""

    run_dir: Path
    repo: str
    """The full Hub repo id."""
    release: str
    """A canvit-pytorch release that reads the scorer, which the card's install line requires."""
    out_dir: Path
    push: bool = False

    def run(self) -> None:
        record = json.loads((self.run_dir / "record.json").read_text())
        scorer = ViewpointScorer.from_pretrained(str(self.run_dir / "scorer"))
        observer = scorer.observer
        for repo in (observer.pretrained_repo, observer.probe_repo):
            assert not Path(repo).exists(), f"the scorer reads {repo}, a local path; publish it first"
        (selected,) = [v for v in record["validations"] if v["step"] == record["selected_step"]]
        training = cards.PolicyTrainingFacts(
            objective=record["objective"], seed=record["config"]["seed"],
            num_glimpses=record["config"]["num_glimpses"], selected_step=record["selected_step"],
            validation_cross_entropy=selected["validation_cross_entropy"],
            record="in config.json, under `training`",
        )
        staged = self.out_dir / self.repo.split("/")[-1]
        shutil.rmtree(staged, ignore_errors=True)
        scorer.save_pretrained(staged)
        _add_training_record(staged, record)
        (staged / "README.md").write_text(cards.policy_card(
            repo=self.repo, observer=observer, candidates=scorer.candidates, training=training,
            release=self.release,
        ))
        reloaded = ViewpointScorer.from_pretrained(str(staged))
        for name, value in reloaded.state_dict().items():
            assert torch.equal(value, scorer.state_dict()[name]), name
        upload(staged, self.repo, push=self.push)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    tyro.extras.subcommand_cli_from_dict({"pretrained": Pretrained, "probe": Probe, "policy": Policy}).run()
