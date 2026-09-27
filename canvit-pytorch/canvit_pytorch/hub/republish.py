"""Republish the checkpoints released in the canvit-pytorch 0.1 format: 0.2 weights and configs, regenerated cards.

A one-time migration, to delete once done. For each released repo it downloads
the 0.1 files, converts them (canvit_pytorch.legacy), writes the card
(canvit_pytorch.hub.cards) and stages config.json, model.safetensors and
README.md in out_dir/<repo name>. A staged checkpoint must reproduce the 0.1
code's outputs, recorded in the reference file, exactly.

Nothing leaves this machine without --push. Pushing first tags each repo's
current commit OLD_FORMAT_REVISION, which 0.1 code can keep loading with
`from_pretrained(repo, revision=OLD_FORMAT_REVISION)`, then uploads the staged files.

    python -m canvit_pytorch.hub.republish --reference ref.pt --out-dir staging
    python -m canvit_pytorch.hub.republish --reference ref.pt --out-dir staging --push
"""

import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, get_args

import torch
import tyro
from huggingface_hub import HfApi, hf_hub_download
from torch import Tensor

from canvit_pytorch import legacy
from canvit_pytorch.hub import cards
from canvit_pytorch.hub.reference import Kind, verify
from canvit_pytorch.hub.repos import (
    FINETUNED_IN1K,
    HUB_ROOT,
    OLD_FORMAT_REVISION,
    PRETRAINED,
    PretrainingDataset,
)
from canvit_pytorch.pretrain.ablations import ABLATIONS
from canvit_pytorch.project import HUB_ORGANIZATION
from canvit_pytorch.teacher import DINOV3_NAMES, DINOV3_PATCH_SIZE, DINOV3_REPOS, DINOv3Variant

log = logging.getLogger(__name__)

OLD_FORMAT_NOTE = f"""
## canvit-pytorch 0.1

This repository's files for canvit-pytorch 0.1 remain at revision `{OLD_FORMAT_REVISION}`:
with `canvit-pytorch<0.2`, pass `revision="{OLD_FORMAT_REVISION}"` to `from_pretrained`.
"""
FINETUNED_TOP1 = (84.5, "C2F, T=21, single run")
"""The released classifier's top-1 accuracy and its conditions, as its 0.1 card reports them."""


def released_repos() -> dict[str, Kind]:
    repos: dict[str, Kind] = {repo: "pretraining" for repo in PRETRAINED.values()}
    repos |= {a.released_repo: "pretraining" for a in ABLATIONS.values()}
    repos[FINETUNED_IN1K] = "classification"
    probes = [m.id for m in HfApi().list_models(author=HUB_ROOT) if m.id.split("/")[1].startswith("probe-ade20k-")]
    assert probes, f"no probe repos under {HUB_ROOT}"
    probe_repos: dict[str, Kind] = {repo: "probe" for repo in sorted(probes)}
    return repos | probe_repos


def _read_json(repo: str, filename: str) -> dict[str, Any]:
    return json.loads(Path(hf_hub_download(repo, filename)).read_text())


def _pretraining_facts(repo: str, old_config: dict[str, Any]) -> cards.PretrainingFacts:
    (grid,) = old_config["canvas_patch_grid_sizes"]
    if (metadata := old_config.get("metadata")) is not None:
        assert metadata["dataset"] in get_args(PretrainingDataset), metadata["dataset"]
        return cards.PretrainingFacts(
            dataset=cast(PretrainingDataset, metadata["dataset"]), teacher_repo=metadata["teacher_repo_id"],
            scene_size_px=metadata["scene_resolution"], glimpse_size_px=16 * metadata["glimpse_grid_size"],
            canvas_grid_size=grid, steps=metadata["step"],
        )
    # The flagship's config records neither; its name does ("-g128px-s512px-in21k-dv3b16-").
    (dataset,) = [d for d, r in PRETRAINED.items() if r == repo]
    dataset = cast(PretrainingDataset, dataset)
    match = re.search(r"-g(\d+)px-s(\d+)px-", repo)
    assert match is not None and "-dv3b16-" in repo, repo
    return cards.PretrainingFacts(
        dataset=dataset, teacher_repo="facebook/dinov3-vitb16-pretrain-lvd1689m",
        scene_size_px=int(match[2]), glimpse_size_px=int(match[1]), canvas_grid_size=grid, steps=None,
    )


def _probe_training(recipe: dict[str, Any]) -> cards.ProbeTraining:
    low, high = recipe["aug_scale_range"]
    return cards.ProbeTraining(
        steps=recipe["max_steps"], batch_size=recipe["batch_size"], peak_lr=recipe["peak_lr"],
        weight_decay=recipe["weight_decay"], warmup_steps=recipe["warmup_steps"], dropout=recipe["dropout"],
        crop_scale_range=(low, high), bfloat16=recipe["amp"],
    )


def _probe_card(repo: str, config: dict[str, Any]) -> str:
    metadata = config["metadata"]
    recipe = metadata["config"]
    if metadata.get("feat_type") == "canvas_hidden":
        assert config["use_ln"], repo
        return cards.canvas_probe_card(repo=repo, facts=cards.CanvasProbeFacts(
            pretrained_repo=recipe["model_repo"],
            scene_size_px=recipe["scene_size"], canvas_grid_size=recipe["canvas_grid"], canvas_dim=config["embed_dim"],
            glimpse_size_px=recipe["glimpse_px"], training_glimpses=recipe["n_timesteps"],
            training_policy="F-IID" if recipe["train_start_full"] else "R-IID", training=_probe_training(recipe),
        ))
    assert not config["use_ln"] and "resolution" in metadata, repo
    variants: list[DINOv3Variant] = [v for v, dinov3_repo in DINOV3_REPOS.items() if dinov3_repo == metadata["model"]]
    assert len(variants) == 1, metadata["model"]
    variant = variants[0]
    return cards.dinov3_probe_card(repo=repo, facts=cards.DINOv3ProbeFacts(
        dinov3_repo=metadata["model"], dinov3_name=DINOV3_NAMES[variant],
        input_size_px=metadata["resolution"], patch_size=DINOV3_PATCH_SIZE, embed_dim=config["embed_dim"],
        training=_probe_training(recipe),
    ))


def _classifier_card(repo: str, config: dict[str, Any]) -> str:
    training = config["training"]
    params = training["params"]
    assert params["chunk_size"] == params["n_glimpses"], "the card states full BPTT"
    details = [
        ("Initialization", (f"[{training['base_checkpoint']}](https://huggingface.co/{training['base_checkpoint']}), "
                            f"its CLS readout fused with the DINOv3 probe `{training['probe_repo']}`")),
        ("Rollouts", f"{params['n_glimpses']} F-IID glimpses of {params['glimpse_size']} px, full BPTT"),
        ("Training", f"{params['epochs']} epochs ({int(params['total_steps']):,} steps), batch size {params['batch_size']}"),
        ("Optimizer", (f"AdamW, learning rate {float(params['lr']):g}, weight decay {float(params['weight_decay']):g}, "
                       f"gradient clipping {params['grad_clip']}")),
        ("Schedule", f"{int(params['warmup_steps']):,}-step linear warmup, then cosine decay to 0"),
        ("Loss", f"cross-entropy at every glimpse, label smoothing {params['label_smoothing']}"),
    ]
    top1, conditions = FINETUNED_TOP1
    return cards.classifier_card(
        repo=repo, pretrained_repo=training["base_checkpoint"], pretraining="in21k", details=details,
        top1_accuracy=top1, conditions=conditions,
    )


def stage(repo: str, kind: Kind, out_dir: Path) -> Path:
    staged = out_dir / repo.split("/")[1]
    shutil.rmtree(staged, ignore_errors=True)
    old_config = _read_json(repo, "config.json")
    if kind == "probe":
        staged.mkdir(parents=True)
        for filename in ("config.json", "model.safetensors"):
            shutil.copyfile(hf_hub_download(repo, filename), staged / filename)
        card = _probe_card(repo, old_config)
    else:
        legacy.convert_checkpoint(repo, staged)
        if kind == "classification":
            card = _classifier_card(repo, old_config)
        else:
            ablation = next((a for a in ABLATIONS.values() if a.released_repo == repo), None)
            card = cards.pretrained_card(repo=repo, facts=_pretraining_facts(repo, old_config), ablation=ablation)
    (staged / "README.md").write_text(card + OLD_FORMAT_NOTE)
    return staged


@dataclass(frozen=True)
class Arguments:
    reference: Path
    """0.1 outputs of every repo on the reference inputs, recorded with the 0.1 code."""
    out_dir: Path
    push: bool = False
    """Tag each repo's current commit OLD_FORMAT_REVISION, then upload the staged files."""
    keep_weights: bool = False
    """Keep each staged model.safetensors after verification (and the push); the full set takes several GB."""


def main(args: Arguments) -> None:
    assert HUB_ROOT == HUB_ORGANIZATION, f"republishing reads and writes the Hub; unset CANVIT_HUB_ROOT (now {HUB_ROOT})"
    references: dict[str, dict[str, Tensor]] = torch.load(args.reference, weights_only=True)
    repos = released_repos()
    assert set(repos) == set(references), f"reference covers {set(references) ^ set(repos)} differently"
    api = HfApi()
    for repo, kind in repos.items():
        staged = stage(repo, kind, args.out_dir)
        verify(staged, kind, references[repo])
        log.info(f"{repo}: staged in {staged}, outputs identical to 0.1")
        if args.push:
            api.create_tag(repo, tag=OLD_FORMAT_REVISION, revision=api.model_info(repo).sha, exist_ok=False)
            api.upload_folder(repo_id=repo, folder_path=staged,
                              commit_message="canvit-pytorch 0.2 checkpoint format and model card")
            log.info(f"{repo}: pushed; 0.1 files remain at revision {OLD_FORMAT_REVISION}")
        if not args.keep_weights:
            (staged / "model.safetensors").unlink()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(tyro.cli(Arguments))
