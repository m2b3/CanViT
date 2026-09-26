"""Each ablation's registry entry reproduces the recipe recorded in its released checkpoint."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest
from huggingface_hub import hf_hub_download

from canvit_pytorch.legacy import convert_canvit_config
from canvit_pytorch.pretrain.ablations import ABLATIONS, AblationSlug
from canvit_pytorch.pretrain.config import PretrainingConfig

BASE = PretrainingConfig(
    shards_dir=Path("shards"), images_dir=Path("images"), validation_dir=Path("val"), dataset="in21k",
    checkpoints_dir=Path("checkpoints"), run_name="run",
)


@pytest.mark.network
@pytest.mark.parametrize("slug", list(ABLATIONS))
def test_matches_released_recipe(slug: AblationSlug) -> None:
    ablation = ABLATIONS[slug]
    released = json.loads(Path(hf_hub_download(ablation.released_repo, "config.json", revision="main")).read_text())
    metadata = released["metadata"]
    history = metadata["training_config_history"]
    recipe = history[max(history)]  # the recipe of the run's last job
    config = ablation.configure(BASE)

    assert asdict(config.model) == convert_canvit_config(released["backbone_name"], released["model_config"])
    assert config.rollout_policies == (
        ("full_then_random",) * recipe["n_full_start_branches"] + ("random",) * recipe["n_random_start_branches"]
    )
    assert config.tbptt_chunk_glimpses == recipe["chunk_size"]
    assert config.stop_probability == pytest.approx(1 - recipe["continue_prob"])
    assert config.enable_teacher_patch_loss == recipe["enable_scene_patches_loss"]
    assert config.enable_teacher_cls_loss == recipe["enable_scene_cls_loss"]
    assert config.warmup_steps == recipe["warmup_steps"]
    assert config.total_steps == metadata["step"]
    assert (config.canvas_grid_size, config.glimpse_size_px) == (recipe["canvas_patch_grid_size"], 16 * recipe["glimpse_grid_size"])
