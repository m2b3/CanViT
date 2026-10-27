"""Regenerate the cards (README.md) of the canvit organization's public repos from each repo's own records and the
results the paper reports, and upload the cards that changed.

    uv run --project canvit-pytorch python -m tools.refresh_cards --out-dir STAGING --live-parity PARITY_REPORT
    uv run --project canvit-pytorch python -m tools.refresh_cards --out-dir STAGING --live-parity PARITY_REPORT --push

Each repo is read at its current main revision: config.json (and provenance.json or manifest.json), its tags, and,
for a classifier, the paper's results in paper/exports/. The new card is staged as OUT_DIR/<repo name>/README.md,
the Hub's as README.hub.md beside it. --push uploads each card that differs from the Hub's, as one commit whose
parent is the revision the card was generated from. The browser exports' cards also need the parity report that
`python -m canvit_pytorch.viz.live parity` wrote for the published graphs.

Left alone, and listed: the learned policies, whose cards come with their republication
(canvit_pytorch.hub.republish_policies), and repos without a record this tool reads.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, get_args

import tyro
from canvit_core.backbone import BACKBONES
from canvit_core.checkpoint import FORMAT
from canvit_pytorch.hub import cards
from canvit_pytorch.hub.publish import probe_card
from canvit_pytorch.hub.repos import (
    OLD_FORMAT_REVISION,
    PRETRAINED,
    RELEASED_GLIMPSE_SIZE_PX,
    RELEASED_SCENE_SIZE_PX,
    PretrainingDataset,
    hub_id,
)
from canvit_pytorch.policies import POLICIES, PolicyName
from canvit_pytorch.pretrain.ablations import ABLATIONS
from canvit_pytorch.project import HUB_ORGANIZATION
from canvit_pytorch.teacher import (
    DINOV3_NAMES,
    DINOV3_PATCH_SIZE,
    DINOV3_REPOS,
    TEACHER_REPO,
    DINOv3Variant,
)
from huggingface_hub import HfApi, hf_hub_download

from tools import model_cards

log = logging.getLogger(__name__)
POLICY_PREFIXES = ("qpolicy-", "pgpolicy-")
RECORD_FILES = ("config.json", "provenance.json", "manifest.json")


@dataclass(frozen=True)
class Repo:
    """A public repo at the revision it was read at."""

    id: str
    revision: str
    tags: frozenset[str]
    files: dict[str, dict[str, Any]]
    """config.json, provenance.json and manifest.json, those the repo has."""

    @property
    def config(self) -> dict[str, Any]:
        return self.files["config.json"]


class NoCard(Exception):
    """This tool does not write the repo's card; the message says why."""


def _json(repo: str, filename: str, revision: str) -> dict[str, Any]:
    return json.loads(Path(hf_hub_download(repo, filename, revision=revision)).read_text())


def _policy_name(policy: str) -> str:
    assert policy in get_args(PolicyName), policy
    return POLICIES[cast(PolicyName, policy)].paper_name


def _top1(entry: dict[str, Any], *, glimpses: int) -> cards.Top1:
    return cards.Top1(glimpses=glimpses, mean=entry["mean"], ci95=(entry["ci_lo"], entry["ci_hi"]))


@dataclass(frozen=True)
class PaperResults:
    """The paper's classification results, as the exporter wrote them to paper/exports/."""

    exports: Path

    def _read(self, name: str) -> dict[str, Any]:
        return json.loads((self.exports / name).read_text())

    def _efficiency_arm(self, repo: str) -> dict[str, Any] | None:
        data = self._read("in1k_efficiency.json")
        arms = [arm for arm in data["canvit"]["arms"].values() if arm["classifier"].get("repo") == repo]
        assert len(arms) <= 1, repo
        return {"policy": data["canvit"]["policy"], "scene_size": data["canvit"]["scene_size"], **arms[0]} if arms else None

    def classification(self, repo: str) -> cards.ClassificationResult:
        """The readings the paper prints for the classifier in `repo`."""
        lpft = self._read("dataset_scale.json")
        for arm in lpft["in1k_lpft"]["arms"].values():
            if arm["model_repo"] == repo:  # the dataset-scale table prints the last glimpse
                last = arm["per_timestep"][-1]
                return cards.ClassificationResult(
                    policy=_policy_name(lpft["_provenance"]["policy"]), scene_size_px=lpft["_provenance"]["scene_size"],
                    canvas_grid_size=lpft["in1k_lpft"]["canvas_grid"], runs=arm["n_runs"],
                    readings=(_top1(last, glimpses=last["t"] + 1),),
                )
        arm = self._efficiency_arm(repo)
        if arm is None:
            raise NoCard(f"{repo}: no classification result in {self.exports}")
        efficiency = cards.ClassificationResult(
            policy=_policy_name(arm["policy"]), scene_size_px=arm["scene_size"], canvas_grid_size=arm["canvas_grid"],
            runs=arm["n_runs"],
            readings=tuple(cards.Top1(glimpses=b["glimpses"], mean=b["top1"], ci95=(b["ci_lo"], b["ci_hi"]))
                           for b in arm["budgets"]),
        )
        if arm["readout"] != "lpft" or arm["canvas_grid"] != self._read("in1k_clf_finetuned.json")["_meta"]["baseline_grid"]:
            return efficiency
        # The released LP-FT classifier: the paper's tables over timesteps (in1k_clf_finetuned) report its best.
        finetuned = self._read("in1k_clf_finetuned.json")
        best = max(finetuned["configs"], key=lambda config: config["best_mean"])
        assert (best["scene_size"], best["canvas_grid"]) == (arm["scene_size"], arm["canvas_grid"]), best
        first = best["per_timestep"][0]
        assert (first["mean"], best["policy"]) == (arm["budgets"][0]["top1"], arm["policy"]), "not the same classifier"
        best_t = best["per_timestep"][best["best_t"]]
        return cards.ClassificationResult(
            policy=_policy_name(best["policy"]), scene_size_px=best["scene_size"], canvas_grid_size=best["canvas_grid"],
            runs=best["n_runs"], readings=(_top1(first, glimpses=1), _top1(best_t, glimpses=best["best_t"] + 1)),
            best_of=best["n_timesteps"],
        )


def _released_dataset(repo: str) -> PretrainingDataset:
    """The pretraining dataset of a released CanViT-B."""
    datasets: list[PretrainingDataset] = [d for d, released in PRETRAINED.items() if released == repo]
    (dataset,) = datasets
    return dataset


def _pretrained_card(repo: Repo) -> str:
    config = repo.config
    ablation = next((a for a in ABLATIONS.values() if a.released_repo == repo.id), None)
    if (metadata := config.get("metadata")) is None:
        # The flagship's config records no training metadata; it has the released viewing geometry and teacher.
        facts = cards.PretrainingFacts(
            dataset=_released_dataset(repo.id), teacher_repo=TEACHER_REPO, scene_size_px=RELEASED_SCENE_SIZE_PX,
            glimpse_size_px=RELEASED_GLIMPSE_SIZE_PX, canvas_grid_size=config["teacher_patch_grid"], steps=None,
        )
    else:
        assert metadata["dataset"] in get_args(PretrainingDataset), metadata["dataset"]
        patch_size = BACKBONES[config["canvit_config"]["backbone_name"]].patch_size
        facts = cards.PretrainingFacts(
            dataset=cast(PretrainingDataset, metadata["dataset"]), teacher_repo=metadata["teacher_repo_id"],
            scene_size_px=metadata["scene_resolution"], glimpse_size_px=patch_size * metadata["glimpse_grid_size"],
            canvas_grid_size=config["teacher_patch_grid"], steps=metadata["step"],
        )
    return cards.pretrained_card(repo=repo.id, facts=facts, ablation=ablation,
                                 old_format=OLD_FORMAT_REVISION in repo.tags)


def _probe_training(metadata: dict[str, Any], crop_size_px: int) -> cards.ProbeTraining:
    """The training of a probe from the record the probe trainers wrote before canvit-pytorch 0.2."""
    recipe = metadata["config"]
    low, high = recipe["aug_scale_range"]
    return cards.ProbeTraining(
        steps=recipe["max_steps"], selected_step=metadata["step"], batch_size=recipe["batch_size"],
        peak_lr=recipe["peak_lr"], weight_decay=recipe["weight_decay"], warmup_steps=recipe["warmup_steps"],
        dropout=recipe["dropout"], crop_size_px=crop_size_px, scale_jitter_range=(low, high),
        flip_probability=recipe["aug_flip_prob"], bfloat16=recipe["amp"],
    )


def _probe_card(repo: Repo) -> str:
    config = repo.config
    old_format = OLD_FORMAT_REVISION in repo.tags
    if "training" in config:  # published by canvit_pytorch.hub.publish probe
        name, card = probe_card(config["training"], embed_dim=config["embed_dim"])
        assert hub_id(name) == repo.id, (name, repo.id)
        return card
    metadata = config["metadata"]
    recipe = metadata["config"]
    if metadata.get("feat_type") == "canvas_hidden":
        assert config["use_ln"], repo.id
        policy: PolicyName = "full_then_random" if recipe["train_start_full"] else "random"
        return cards.canvas_probe_card(repo=repo.id, old_format=old_format, facts=cards.CanvasProbeFacts(
            pretrained_repo=recipe["model_repo"], scene_size_px=recipe["scene_size"],
            canvas_grid_size=recipe["canvas_grid"], canvas_dim=config["embed_dim"], glimpse_size_px=recipe["glimpse_px"],
            training_glimpses=recipe["n_timesteps"], training_policy=POLICIES[policy].paper_name,
            training=_probe_training(metadata, crop_size_px=recipe["scene_size"]),
        ))
    assert not config["use_ln"] and "resolution" in metadata, repo.id
    variants: list[DINOv3Variant] = [v for v, dinov3_repo in DINOV3_REPOS.items() if dinov3_repo == metadata["model"]]
    (variant,) = variants
    return cards.dinov3_probe_card(repo=repo.id, old_format=old_format, facts=cards.DINOv3ProbeFacts(
        dinov3_repo=metadata["model"], dinov3_name=DINOV3_NAMES[variant], input_size_px=metadata["resolution"],
        patch_size=DINOV3_PATCH_SIZE, embed_dim=config["embed_dim"],
        training=_probe_training(metadata, crop_size_px=recipe["image_size"]),
    ))


def _link(repo: str) -> str:
    return f"[{repo}](https://huggingface.co/{repo})"


def _lpft_rows_tpu_xla(training: dict[str, Any]) -> list[tuple[str, str]]:
    """The fine-tuning recipe from the record of the PyTorch/XLA trainer (canvit_pytorch.specialize.in1k_tpu), whose
    parameters it stores as strings."""
    params = training["params"]
    assert params["use_pretrained_probe"] == "true" and params["curr_step"] == params["total_steps"], params
    glimpses, chunk = int(params["n_glimpses"]), int(params["chunk_size"])
    return [
        ("Initialization", (f"{_link(training['base_checkpoint'])}, its CLS readout fused with the DINOv3 probe "
                            f"{_link(hub_id(training['probe_repo']))}")),
        ("Rollouts", (f"{glimpses} F-IID glimpses of {params['glimpse_size']} px"
                      + (", full backpropagation through time" if chunk >= glimpses else f", truncated every {chunk}"))),
        ("Training", f"{params['epochs']} epochs ({int(params['total_steps']):,} steps), batch size {params['batch_size']}"),
        ("Optimizer", (f"AdamW, learning rate {float(params['lr']):g}, weight decay {float(params['weight_decay']):g}, "
                       f"gradient clipping {float(params['grad_clip']):g}")),
        ("Schedule", f"{int(params['warmup_steps']):,}-step linear warmup, then cosine decay to 0"),
        ("Loss", f"cross-entropy after every glimpse, label smoothing {float(params['label_smoothing']):g}"),
        ("Precision", "float32" if params["amp_enabled"] == "false" else "bfloat16 autocast"),
    ]


def _lpft_rows_nnx(training: dict[str, Any]) -> list[tuple[str, str]]:
    """The fine-tuning recipe from the record of the JAX/Flax NNX exports."""
    grid = training["canvas_grid_size"]
    return [
        ("Initialization", (f"{_link(training['base_checkpoint'])}, its CLS readout fused with the DINOv3 probe "
                            f"{_link(training['fused_probe'])}")),
        ("Rollouts", (f"{training['glimpses']} glimpses of {training['glimpse_size_px']} px on "
                      f"{training['scene_size_px']} px scenes, {grid} × {grid} canvas")),
        ("Training", f"{training['checkpoint_step']:,} of {training['total_steps']:,} steps, batch size {training['batch_size']}"),
        ("Optimizer", (f"AdamW, learning rate {training['learning_rate']:g}, weight decay {training['weight_decay']:g}, "
                       f"gradient clipping {training['grad_clip']:g}")),
        ("Schedule", f"{training['warmup_steps']:,}-step linear warmup, then cosine decay to 0"),
        ("Loss", f"cross-entropy after every glimpse, label smoothing {training['label_smoothing']:g}"),
        ("Trainer", training["trainer"]),
    ]


def _classifier_card(repo: Repo, results: PaperResults) -> str:
    training = repo.config["training"]
    rows = _lpft_rows_tpu_xla(training) if "params" in training else _lpft_rows_nnx(training)
    base = training["base_checkpoint"]
    return cards.classifier_card(
        repo=repo.id, pretrained_repo=base, pretraining=_released_dataset(base), training=rows,
        result=results.classification(repo.id), old_format=OLD_FORMAT_REVISION in repo.tags,
    )


def _dinov3_classification_probe_card(repo: Repo) -> str:
    config = repo.config
    trial, metadata, scores = config["trial_params"], config["config_metadata"], config["val_results"]
    assert trial["optimizer"] == "adamw", trial["optimizer"]
    return cards.dinov3_classification_probe_card(repo=repo.id, facts=cards.DINOv3ClassificationProbeFacts(
        variant=metadata["model_name"].removeprefix("dinov3_"), image_size_px=metadata["image_size"],
        in_features=config["in_features"], num_classes=config["out_features"], top1=scores["top1"],
        real_top1=scores["real_top1"], top5=scores["top5"], steps=trial["total_steps"],
        batch_size=trial["batch_size"], peak_lr=trial["peak_lr"], weight_decay=trial["weight_decay"],
        warmup_steps=trial["warmup_steps"],
    ))


def _native_card(repo: Repo, results: PaperResults) -> str:
    config, provenance = repo.config, repo.files["provenance.json"]
    backend = config["backend"]
    source = model_cards.Source(provenance["source"]["repo"], provenance["source"]["revision"])
    if config["model"] == "SegmentationProbe":
        recipe = _json(source.repo, "config.json", source.revision)["metadata"]["config"]
        return model_cards.probe_card(model_cards.NativeProbe(
            repo=repo.id, backend=backend, record=config, pretrained_native_repo=provenance["pretrained_native_repo"],
            geometry=model_cards.Geometry(recipe["scene_size"], recipe["glimpse_px"], recipe["canvas_grid"]),
            source=source,
        ))
    validation = provenance["validation"]
    geometry = model_cards.Geometry(scene_size_px=validation["scene_size_px"], glimpse_size_px=validation["glimpse_size_px"],
                                    canvas_grid_size=validation["canvas_grid_size"])
    training = None
    if source.repo == repo.id:
        # Converted from this repo's own earlier revision: no PyTorch repo of its own states its training and result.
        record = _json(repo.id, "config.json", source.revision)["training"]
        base, result = record["base_checkpoint"], results.classification(repo.id)
        grid = result.canvas_grid_size
        training = model_cards.NativeTraining(
            title=f"CanViT-B fine-tuned on ImageNet-1k, {grid} × {grid} canvas",
            summary=(f"{_link(base)} fine-tuned end to end for ImageNet-1k classification by linear probing then "
                     f"fine-tuning (LP-FT) with its canvas at {grid} × {grid}, in the checkpoint format of "
                     f"canvit-{backend}."),
            rows=[("Fine-tuning", (f"LP-FT from {_link(base)}, {int(record['checkpoint_step']):,} steps, with JAX/Flax "
                                   "NNX on Cloud TPU")),
                  ("Evaluated", (f"the canvit-pytorch 0.1 export at revision `{source.revision[:8]}`, which this "
                                 "conversion reproduces (provenance.json)"))],
            results=result.section(), base_model=base, datasets=(cards.IMAGENET_1K_HUB_ID,),
        )
    return model_cards.model_card(model_cards.NativeModel(
        repo=repo.id, backend=backend, record=config, geometry=geometry, source=source, training=training,
    ))


def _live_card(repo: Repo, parity: dict[str, Any]) -> str:
    manifest = repo.files["manifest.json"]
    role = "probe" if "readout" in manifest else "canvit"
    assert parity["graphs_sha256"][role] == manifest["graph"]["sha256"], f"{repo.id}: the parity report is of other graphs"
    other = {"canvit": "probe", "probe": "canvit"}[role]
    (partner,) = [
        m.id for m in HfApi().list_models(author=HUB_ORGANIZATION, search="onnx-fp32")
        if m.id != repo.id and _json(m.id, "manifest.json", "main")["graph"]["sha256"] == parity["graphs_sha256"][other]
    ]
    if role == "canvit":
        return cards.live_canvit_card(repo=repo.id, probe_repo=partner, manifest=manifest, parity=parity)
    return cards.live_probe_card(repo=repo.id, canvit_repo=partner, manifest=manifest, parity=parity)


def card(repo: Repo, results: PaperResults, parity: dict[str, Any] | None) -> str:
    name = repo.id.split("/")[1]
    if name.startswith(POLICY_PREFIXES):
        raise NoCard("a learned policy: its card comes with its republication (canvit_pytorch.hub.republish_policies)")
    if "manifest.json" in repo.files:
        if parity is None:
            raise NoCard("a browser export: pass --live-parity")
        return _live_card(repo, parity)
    if "config.json" not in repo.files:
        raise NoCard("no config.json")
    config = repo.config
    if config.get("format") == FORMAT:
        return _native_card(repo, results)
    if "teacher_dim" in config:
        return _pretrained_card(repo)
    if "use_ln" in config:
        return _probe_card(repo)
    if "n_classes" in config and "training" in config:
        return _classifier_card(repo, results)
    if "config_metadata" in config and "val_results" in config:
        return _dinov3_classification_probe_card(repo)
    raise NoCard(f"no card for a config with keys {sorted(config)}")


@dataclass(frozen=True)
class RefreshCards:
    out_dir: Path
    exports: Path = Path("paper/exports")
    """The paper's exported results, read for the classifiers' cards."""
    live_parity: Path | None = None
    """parity/report.json of the published browser export."""
    only: tuple[str, ...] = ()
    """Repo names to refresh; every public repo when empty."""
    push: bool = False

    def run(self) -> None:
        api = HfApi()
        results = PaperResults(self.exports)
        parity = json.loads(self.live_parity.read_text()) if self.live_parity else None
        changed, skipped = [], []
        for model in api.list_models(author=HUB_ORGANIZATION):
            info = api.model_info(model.id)
            name = model.id.split("/")[1]
            if info.private or (self.only and name not in self.only):
                continue
            assert info.sha is not None and info.siblings is not None, model.id
            present = {s.rfilename for s in info.siblings}
            files = {f: _json(model.id, f, info.sha) for f in RECORD_FILES if f in present}
            repo = Repo(id=model.id, revision=info.sha, files=files,
                        tags=frozenset(t.name for t in api.list_repo_refs(model.id).tags))
            try:
                new = card(repo, results, parity)
            except NoCard as reason:
                skipped.append((model.id, str(reason)))
                continue
            staged = self.out_dir / name
            staged.mkdir(parents=True, exist_ok=True)
            (staged / "README.md").write_text(new)
            old = Path(hf_hub_download(model.id, "README.md", revision=info.sha)).read_text() if "README.md" in present else ""
            (staged / "README.hub.md").write_text(old)
            (staged / "revision").write_text(info.sha + "\n")
            if new != old:
                changed.append((model.id, info.sha, staged))
        for repo_id, reason in skipped:
            log.info(f"left alone: {repo_id}: {reason}")
        log.info(f"{len(changed)} cards differ from the Hub's, staged in {self.out_dir}")
        if not self.push:
            return
        for repo_id, revision, staged in changed:
            commit = api.upload_file(
                path_or_fileobj=staged / "README.md", path_in_repo="README.md", repo_id=repo_id,
                commit_message="Regenerate the model card", parent_commit=revision,
            )
            log.info(f"{repo_id}: {commit.commit_url}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    tyro.cli(RefreshCards).run()
