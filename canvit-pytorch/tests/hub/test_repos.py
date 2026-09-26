import json

import pytest

import canvit_pytorch
from canvit_pytorch import CanViTForPretraining
from canvit_pytorch.hub.repos import FLAGSHIP, hub_repo, released_ade20k_probe, released_dinov3_ade20k_probe


def test_released_names_are_the_published_ones():
    assert FLAGSHIP == hub_repo("canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02")
    assert released_ade20k_probe("in21k", scene_size_px=512, canvas_grid_size=64) == hub_repo("probe-ade20k-40k-s512-c64-in21k")
    assert released_dinov3_ade20k_probe("dv3b", input_size_px=512) == hub_repo("probe-ade20k-40k-dv3b-512px")


def test_a_renamed_class_names_its_replacement():
    with pytest.raises(AttributeError, match="is now CanViTForPretraining"):
        _ = canvit_pytorch.CanViTForPretrainingHFHub


def test_a_0_1_checkpoint_fails_with_the_upgrade_hint(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"backbone_name": "vitb16", "model_config": {}, "teacher_dim": 768}))
    with pytest.raises(ValueError, match="canvit-pytorch 0.1 checkpoint"):
        CanViTForPretraining.from_pretrained(str(tmp_path))
