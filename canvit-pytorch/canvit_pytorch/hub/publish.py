"""Publish a trained checkpoint or probe to the Hub: weights, config.json with the training record, and its card.

Each command stages the repo in out_dir/<repo name>, loads the staged files
with the class users will load them with, and uploads only with --push, to a
new private repo (made public on the Hub once reviewed).

    python -m canvit_pytorch.hub.publish pretrained --checkpoint RUN/step-2000000.pt --repo NAME --out-dir staging
    python -m canvit_pytorch.hub.publish probe --run-dir PROBE_RUN --out-dir staging
    python -m canvit_pytorch.hub.publish classifier --export-dir EXPORT --reference ref.pt --repo NAME ... --out-dir staging
"""

import dataclasses
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import tyro
from huggingface_hub import HfApi

from canvit_pytorch import legacy
from canvit_pytorch.hub import cards, reference
from canvit_pytorch.hub.repos import (
    DINOV3_PROBE_MODEL_NAMES,
    DINOV3_VITB16_IN1K_PROBE,
    PRETRAINED,
    RELEASED_CANVAS_GRID_SIZE,
    RELEASED_GLIMPSE_SIZE_PX,
    RELEASED_SCENE_SIZE_PX,
    PretrainingDataset,
    ade20k_probe_name,
    dinov3_ade20k_probe_name,
)
from canvit_pytorch.model.pretraining import CanViTForPretraining
from canvit_pytorch.policies import POLICIES, PolicyName
from canvit_pytorch.pretrain import checkpoint
from canvit_pytorch.pretrain.ablations import ABLATIONS, AblationSlug
from canvit_pytorch.probes import SegmentationProbe
from canvit_pytorch.project import HUB_ORGANIZATION
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
        staged = self.out_dir / self.repo.split("/")[-1]
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


def _hub_id(name: str) -> str:
    """Published repos go to the Hub organization, also when CANVIT_HUB_ROOT points at a local mirror."""
    return f"{HUB_ORGANIZATION}/{name}"


def _released_checkpoint(pretrained_repo: str) -> tuple[str, str]:
    """(Hub id, name in probe names) of a released checkpoint, given its Hub id or a local mirror directory."""
    names = {Path(repo).name: dataset for dataset, repo in PRETRAINED.items()}
    names |= {Path(a.released_repo).name: a.probe_model_name for a in ABLATIONS.values()}
    name = Path(pretrained_repo).name
    assert name in names, f"{pretrained_repo} is not a released checkpoint; its probes have no name yet"
    return _hub_id(name), names[name]


@dataclass(frozen=True)
class Probe:
    """A probe-training run directory (canvit_pytorch.specialize.ade20k) as a SegmentationProbe repo, named by
    the probe-name grammar of canvit_pytorch.hub.repos."""

    run_dir: Path
    out_dir: Path
    push: bool = False

    def run(self) -> None:
        record = json.loads((self.run_dir / "record.json").read_text())
        setup, training = record["config"], record["config"]["training"]
        probe = SegmentationProbe.from_pretrained(str(self.run_dir / record["probe_dir"]))
        low, high = training["scale_jitter_range"]
        probe_training = cards.ProbeTraining(
            steps=record["steps_trained"], batch_size=training["batch_size"], peak_lr=training["peak_lr"],
            weight_decay=training["weight_decay"], warmup_steps=training["warmup_steps"],
            dropout=training["dropout"], crop_scale_range=(low, high), bfloat16=training["feature_dtype"] == "bfloat16",
        )
        match record["config_type"]:
            case "CanvasProbeConfig":
                pretrained_repo, model_name = _released_checkpoint(setup["pretrained_repo"])
                name = ade20k_probe_name(
                    model_name, scene_size_px=training["scene_size_px"],
                    canvas_grid_size=setup["canvas_grid_size"], steps=record["steps_trained"],
                )
                card = cards.canvas_probe_card(repo=_hub_id(name), facts=cards.CanvasProbeFacts(
                    pretrained_repo=pretrained_repo, scene_size_px=training["scene_size_px"],
                    canvas_grid_size=setup["canvas_grid_size"], canvas_dim=probe.embed_dim,
                    glimpse_size_px=setup["glimpse_size_px"], training_glimpses=setup["num_glimpses"],
                    training_policy=POLICIES[cast(PolicyName, setup["training_policy"])].paper_name,
                    training=probe_training,
                ))
            case "DINOv3ProbeConfig":
                variant = cast(DINOv3Variant, setup["variant"])
                name = dinov3_ade20k_probe_name(
                    DINOV3_PROBE_MODEL_NAMES[variant], input_size_px=setup["input_size_px"], steps=record["steps_trained"],
                )
                card = cards.dinov3_probe_card(repo=_hub_id(name), facts=cards.DINOv3ProbeFacts(
                    dinov3_repo=DINOV3_REPOS[variant], dinov3_name=DINOV3_NAMES[variant],
                    input_size_px=setup["input_size_px"], patch_size=DINOV3_PATCH_SIZE, embed_dim=probe.embed_dim,
                    training=probe_training,
                ))
            case other:
                raise ValueError(f"{self.run_dir}: unknown probe configuration {other!r}")
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
        upload(staged, _hub_id(name), push=self.push)


@dataclass(frozen=True)
class LpftRecipe:
    """How the classifier was fine-tuned: linear probing then fine-tuning (LP-FT) with F-IID rollouts, full BPTT and
    cross-entropy at every glimpse, at the released geometry. The card states these values."""

    trainer: str
    """Where it was trained, e.g. "JAX/Flax NNX on Cloud TPU, exported to PyTorch"."""
    glimpses: int
    batch_size: int
    total_steps: int
    checkpoint_step: int
    """The step of the published weights; the last checkpoint saved can precede the schedule's end."""
    warmup_steps: int
    learning_rate: float
    weight_decay: float
    grad_clip: float
    label_smoothing: float
    run_id: str


@dataclass(frozen=True)
class Classifier:
    """An ImageNet-1k classifier exported in the canvit-pytorch 0.1 format by a trainer outside this package, as a
    CanViTForImageClassification repo. Its 0.2 conversion must reproduce the 0.1 code's outputs, which
    scripts/record_0_1_outputs.py records."""

    export_dir: Path
    """config.json and model.safetensors in the 0.1 format; its training record names the base checkpoint."""
    reference: Path
    repo: str
    """The full Hub repo id."""
    pretraining: PretrainingDataset
    top1_accuracy: float
    accuracy_conditions: str
    """How the accuracy was measured, e.g. "C2F, T=21, mean over 11 policy seeds"."""
    recipe: LpftRecipe
    out_dir: Path
    push: bool = False

    def run(self) -> None:
        recorded = torch.load(self.reference, weights_only=True)
        assert recorded["inputs"] == reference.inputs(), "the reference was recorded on other inputs"
        outputs = recorded["models"][str(self.export_dir.resolve())]
        base = json.loads((self.export_dir / "config.json").read_text())["training"]["base_checkpoint"]
        assert base == PRETRAINED[self.pretraining], (base, self.pretraining)
        staged = self.out_dir / self.repo.split("/")[-1]
        shutil.rmtree(staged, ignore_errors=True)
        legacy.convert_checkpoint(str(self.export_dir), staged)
        reference.verify(staged, "classification", outputs)
        r = self.recipe
        config_path = staged / "config.json"
        config = json.loads(config_path.read_text())
        config["training"] = {"base_checkpoint": base, "fused_probe": DINOV3_VITB16_IN1K_PROBE,
                              "scene_size_px": RELEASED_SCENE_SIZE_PX, "glimpse_size_px": RELEASED_GLIMPSE_SIZE_PX,
                              "canvas_grid_size": RELEASED_CANVAS_GRID_SIZE, **dataclasses.asdict(r)}
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        details = [
            ("Initialization", (f"[{base}](https://huggingface.co/{base}), its CLS readout fused with the DINOv3 probe "
                                f"`{DINOV3_VITB16_IN1K_PROBE}`")),
            ("Rollouts", (f"{r.glimpses} F-IID glimpses of {RELEASED_GLIMPSE_SIZE_PX} px on {RELEASED_SCENE_SIZE_PX} px "
                          f"scenes, {RELEASED_CANVAS_GRID_SIZE} × {RELEASED_CANVAS_GRID_SIZE} canvas, full BPTT")),
            ("Training", f"{r.checkpoint_step:,} of {r.total_steps:,} steps, batch size {r.batch_size}"),
            ("Optimizer", (f"AdamW, learning rate {r.learning_rate:g}, weight decay {r.weight_decay:g}, gradient "
                           f"clipping {r.grad_clip:g}")),
            ("Schedule", f"{r.warmup_steps:,}-step linear warmup, then cosine decay to 0"),
            ("Loss", f"cross-entropy at every glimpse, label smoothing {r.label_smoothing:g}"),
            ("Trainer", r.trainer),
        ]
        (staged / "README.md").write_text(cards.classifier_card(
            repo=self.repo, pretrained_repo=base, pretraining=self.pretraining, details=details,
            top1_accuracy=self.top1_accuracy, conditions=self.accuracy_conditions,
        ))
        upload(staged, self.repo, push=self.push)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    tyro.extras.subcommand_cli_from_dict({"pretrained": Pretrained, "probe": Probe, "classifier": Classifier}).run()
