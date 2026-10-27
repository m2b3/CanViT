"""canvit-nnx restates facts whose home is canvit-pytorch, which it cannot import; these tests hold the two together."""

import dataclasses
import json
from typing import get_args

import pytest
from canvit_core.checkpoint import hub_file
from canvit_pytorch.benchmarks import imagenet as torch_imagenet
from canvit_pytorch.hub import repos as torch_repos
from canvit_pytorch.policies import MIN_SCALE as TORCH_MIN_SCALE

from canvit_nnx import releases
from canvit_nnx.policies import MIN_SCALE
from canvit_nnx.specialize.in1k import data
from canvit_nnx.specialize.in1k.train import Recipe


def test_released_facts_match_canvit_pytorch() -> None:
    assert get_args(releases.PretrainingDataset) == get_args(torch_repos.PretrainingDataset)
    assert releases.PRETRAINED == {dataset: f"{repo}-nnx" for dataset, repo in torch_repos.PRETRAINED.items()}
    assert releases.DINOV3_VITB16_IN1K_PROBE == torch_repos.DINOV3_VITB16_IN1K_PROBE
    assert releases.RELEASED_GLIMPSE_SIZE_PX == torch_repos.RELEASED_GLIMPSE_SIZE_PX
    assert releases.RELEASED_SCENE_SIZE_PX == torch_repos.RELEASED_SCENE_SIZE_PX
    assert releases.RELEASED_CANVAS_GRID_SIZE == torch_repos.RELEASED_CANVAS_GRID_SIZE
    assert MIN_SCALE == TORCH_MIN_SCALE
    assert data.NUM_CLASSES == torch_imagenet.NUM_CLASSES
    assert data.NUM_TRAIN_IMAGES == torch_imagenet.NUM_TRAIN_IMAGES
    assert data.NUM_VALIDATION_IMAGES == torch_imagenet.NUM_VALIDATION_IMAGES


@pytest.mark.network
@pytest.mark.parametrize("pretraining_dataset", get_args(releases.PretrainingDataset))
def test_the_default_recipe_is_that_of_the_released_jax_fine_tuning(
    pretraining_dataset: releases.PretrainingDataset,
) -> None:
    record = json.loads(hub_file(torch_repos.FINETUNED_IN1K_NNX[pretraining_dataset], "config.json").read_text())
    trained = record["training"]
    recipe = dataclasses.replace(Recipe(), pretraining_dataset=pretraining_dataset)
    assert trained["base_checkpoint"] == torch_repos.PRETRAINED[recipe.pretraining_dataset]
    assert trained["fused_probe"] == releases.DINOV3_VITB16_IN1K_PROBE
    assert (trained["scene_size_px"], trained["glimpse_size_px"]) == (
        releases.RELEASED_SCENE_SIZE_PX, releases.RELEASED_GLIMPSE_SIZE_PX,
    )
    assert {
        "batch_size": recipe.batch_size,
        "total_steps": recipe.total_steps,
        "warmup_steps": recipe.warmup_steps,
        "learning_rate": recipe.peak_lr,
        "weight_decay": recipe.weight_decay,
        "grad_clip": recipe.max_grad_norm,
        "label_smoothing": recipe.label_smoothing,
        "glimpses": recipe.num_glimpses,
        "canvas_grid_size": recipe.canvas_grid_size,
    } == {name: trained[name] for name in (
        "batch_size", "total_steps", "warmup_steps", "learning_rate", "weight_decay", "grad_clip", "label_smoothing",
        "glimpses", "canvas_grid_size",
    )}
