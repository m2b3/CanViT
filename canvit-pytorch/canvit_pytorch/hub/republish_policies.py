"""Republish the learned viewing policies in the format of canvit_pytorch.rl: converted weights and config,
regenerated cards.

A one-time migration, to delete once done. The policies were published by the code that trained them, whose
scorer names its weights differently, stores its candidates' centers and records redundant config keys. For each
policy repo of the Hub organization, this stages config.json, model.safetensors and README.md in
out_dir/<repo name>; the repo's training record, training_run/, stays as it is. Every staged weight equals the
original weight it renames, and the card's training facts come from training_run/.

Nothing leaves this machine without --push, which comes before the first canvit-pytorch release with
canvit_pytorch.rl (CHANGELOG.md): once every repo is staged, it tags the commit each was staged from
ORIGINAL_FORMAT_REVISION and uploads the staged files on top of that commit.

    python -m canvit_pytorch.hub.republish_policies --release VERSION --out-dir staging
    python -m canvit_pytorch.hub.republish_policies --release VERSION --out-dir staging --push
"""

import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import tyro
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file
from torch import Tensor

from canvit_pytorch.hub import cards
from canvit_pytorch.hub.repos import HUB_ROOT
from canvit_pytorch.project import HUB_ORGANIZATION
from canvit_pytorch.rl import ViewpointScorer
from canvit_pytorch.rl.candidates import CandidateViewpoints
from canvit_pytorch.rl.observer import flagship_observer

log = logging.getLogger(__name__)

ORIGINAL_FORMAT_REVISION = "original-format"
ORIGINAL_FEATURE_GROUPS = ["ent", "ent_delta", "cos_prev", "cos_init", "ln_feat", "feat_delta"]
ORIGINAL_OBJECTIVES = {"q": "QRegression", "pg": "PolicyGradient"}
ORIGINAL_FORMAT_NOTE = f"""
## Original format

The files in the format of the code that trained this policy remain at revision `{ORIGINAL_FORMAT_REVISION}`;
canvit-pytorch reads those of `main`.
"""

# Original weight names → canvit_pytorch.rl's, applied in order; the stages' inner names are renamed apart.
RULES: list[tuple[str, str]] = [
    (r"^frontend\.bn\.", "feature_fusion.input_norm."),
    (r"^frontend\.proj\.", "feature_fusion.group_projections."),
    (r"^frontend\.norm\.", "feature_fusion.group_norms."),
    (r"^frontend\.gate$", "feature_fusion.group_gates"),
    (r"^frontend\.mlp\.norm\.", "feature_fusion.mlp.0."),
    (r"^frontend\.mlp\.net\.0\.", "feature_fusion.mlp.1."),
    (r"^frontend\.mlp\.net\.2\.", "feature_fusion.mlp.3."),
    (r"^frontend\.out\.", "feature_fusion.output_projection."),
    (r"^head\.1\.", "score_head.1."),
    (r"^scale_emb$", "scale_embeddings"),
    (r"^vhead\.", "value_head."),
]
STAGES = {"enc": "unet.encoder_stages.", "dec": "unet.decoder_stages.", "head": "score_head."}
BLOCK_RULES: list[tuple[str, str]] = [
    (r"^conv_dw\.", "depthwise_conv."), (r"^gamma$", "layer_scale"), (r"^mlp\.fc1\.", "fc1."), (r"^mlp\.fc2\.", "fc2."),
    (r"^mlp\.grn\.", "global_response_norm."),
]


def converted_key(key: str) -> str:
    """The canvit_pytorch.rl name of an original scorer weight. An original stage is a Sequential of the decoder's
    projection, then the blocks; the score head's stage is its module 0."""
    if match := re.fullmatch(r"(enc|dec|head)\.(\d+)\.(\d+)\.(.+)", key):
        kind, stage, index, rest = match[1], int(match[2]), int(match[3]), match[4]
        prefix = STAGES[kind] + f"{stage}."
        if kind == "dec" and index == 0:
            return f"{prefix}projection.{rest}"
        for pattern, replacement in BLOCK_RULES:
            rest = re.sub(pattern, replacement, rest)
        return f"{prefix}blocks.{index - (kind == 'dec')}.{rest}"
    for pattern, replacement in RULES:
        if re.search(pattern, key):
            return re.sub(pattern, replacement, key)
    raise ValueError(f"no rule renames the original weight {key}")


def converted_scorer(config: dict[str, Any], manifest: dict[str, Any], weights: dict[str, Tensor]) -> ViewpointScorer:
    """The original scorer in canvit_pytorch.rl, every weight equal to the original it renames."""
    assert config.pop("groups") == ORIGINAL_FEATURE_GROUPS and config.pop("n_scale") == len(config["scales"]), config
    run = manifest["config"]
    observer = flagship_observer(canvas_grid_size=run["canvas_grid"])
    assert (run["model_repo"], run["scene_size"], run["glimpse_px"]) == (
        observer.pretrained_repo, observer.scene_size_px, observer.glimpse_size_px,
    ), run
    assert config.pop("dueling") == (manifest["kind"] == "q"), manifest["kind"]
    candidates = CandidateViewpoints(tuple(config.pop("scales")), config.pop("centers_per_axis"))
    scorer = ViewpointScorer(
        observer=observer, canvas_dim=config.pop("canvas_dim"), candidates=candidates, width=config.pop("width"),
        blocks_per_stage=config.pop("block_layers"), dueling=manifest["kind"] == "q",
    )
    assert not config, f"unconverted config keys {config}"
    assert torch.equal(weights.pop("sb_grids"), scorer.candidate_centers_xy), "the stored centers are the candidates'"
    converted = {converted_key(key): value for key, value in weights.items()}
    assert len(converted) == len(weights), "two original weights renamed alike"
    scorer.load_state_dict(converted, strict=True)
    for key, value in weights.items():
        assert torch.equal(scorer.state_dict()[converted_key(key)], value), key
    return scorer


def training_facts(
    manifest: dict[str, Any], metrics: list[dict[str, Any]], original_card: str,
) -> cards.PolicyTrainingFacts:
    """From training_run/: the published scorer is that of lowest mean validation cross-entropy after the chosen
    glimpses, which the original card names."""
    num_glimpses = manifest["config"]["train_horizon"] + 1
    validations = {
        row["step"]: [row[f"val_ce_t{t}"] for t in range(num_glimpses)] for row in metrics if "val_ce_t1" in row
    }
    selected_step = min(validations, key=lambda step: sum(validations[step][1:]))
    stated = re.search(r", step (\d+)\n", original_card)
    assert stated is not None and int(stated[1]) == selected_step, (stated, selected_step)
    return cards.PolicyTrainingFacts(
        objective=ORIGINAL_OBJECTIVES[manifest["kind"]], seed=manifest["config"]["seed"], num_glimpses=num_glimpses,
        selected_step=selected_step, validation_cross_entropy=validations[selected_step],
        record="under `training_run/`, as the original training code wrote it",
    )


def policy_repos() -> list[str]:
    models = HfApi().list_models(author=HUB_ORGANIZATION)
    repos = sorted(model.id for model in models if re.match(r"(q|pg)policy-ade20k-", model.id.split("/")[1]))
    assert repos, f"no policy repos under {HUB_ORGANIZATION}"
    return repos


def stage(repo: str, out_dir: Path, *, revision: str, release: str) -> Path:
    """Convert the policy of `repo` at the Hub commit `revision` into out_dir/<repo name>, verified by reloading."""

    def download(filename: str) -> Path:
        return Path(hf_hub_download(repo, filename, revision=revision))

    manifest = json.loads(download("training_run/manifest.json").read_text())
    metrics = [json.loads(line) for line in download("training_run/metrics.jsonl").read_text().splitlines()]
    scorer = converted_scorer(
        json.loads(download("config.json").read_text()), manifest, load_file(download("model.safetensors")),
    )
    training = training_facts(manifest, metrics, download("README.md").read_text())
    staged = out_dir / repo.split("/")[1]
    shutil.rmtree(staged, ignore_errors=True)
    scorer.save_pretrained(staged)
    (staged / "README.md").write_text(cards.policy_card(
        repo=repo, observer=scorer.observer, candidates=scorer.candidates, training=training,
        release=release,
    ) + ORIGINAL_FORMAT_NOTE)
    reloaded = ViewpointScorer.from_pretrained(str(staged))
    assert reloaded.observer == scorer.observer and reloaded.candidates == scorer.candidates, staged
    for name, value in scorer.state_dict().items():
        assert torch.equal(reloaded.state_dict()[name], value), name
    return staged


@dataclass(frozen=True)
class Arguments:
    release: str
    """The first canvit-pytorch release with canvit_pytorch.rl, which the cards' install line requires."""
    out_dir: Path
    push: bool = False
    """Tag the commit each repo was staged from ORIGINAL_FORMAT_REVISION, then upload the staged files on it."""


def main(args: Arguments) -> None:
    assert HUB_ROOT == HUB_ORGANIZATION, f"republishing reads the Hub; unset CANVIT_HUB_ROOT (now {HUB_ROOT})"
    api = HfApi()
    # Every repo is staged and verified before any is pushed, from the commit its tag will name.
    originals = {repo: api.model_info(repo).sha for repo in policy_repos()}
    staged = {}
    for repo, revision in originals.items():
        assert revision, f"no Hub commit for {repo}"
        staged[repo] = stage(repo, args.out_dir, revision=revision, release=args.release)
        log.info(f"{repo}: staged from {revision} in {staged[repo]}")
    if not args.push:
        return
    for repo, revision in originals.items():
        api.create_tag(repo, tag=ORIGINAL_FORMAT_REVISION, revision=revision, exist_ok=False)
        api.upload_folder(
            repo_id=repo, folder_path=staged[repo], commit_message="canvit_pytorch.rl's format, its card",
            parent_commit=revision,
        )
        log.info(f"{repo}: pushed; the original files remain at revision {ORIGINAL_FORMAT_REVISION}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(tyro.cli(Arguments))
