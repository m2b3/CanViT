import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from canvit_core import CanViTConfig
from flax import nnx

from canvit_nnx.detect import train
from canvit_nnx.detect.model import CanViTForObjectDetection
from canvit_nnx.model import CanViTForPretraining


def _metrics(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]


def test_an_interrupted_run_resumes_as_if_uninterrupted(
    small_config: CanViTConfig, coco_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(train, "NUM_TRAINING_IMAGES", 7)
    pretrained = tmp_path / "pretrained"
    CanViTForPretraining(canvit_config=small_config, teacher_dim=8, teacher_patch_grid=2, rngs=nnx.Rngs(0)).save_pretrained(
        pretrained
    )
    recipe = train.Recipe(
        pretrained=str(pretrained), compute_dtype="float32", batch_size=2, num_glimpses=2, head_width=8, epochs=2,
        canvas_grid_size=4,
    )
    assert recipe.total_steps == 7

    def run(run_dir: Path, stop_after_step: int | None) -> None:
        train.train(train.DetectionTrainingConfig(
            data_dir=coco_dir, run_dir=run_dir, recipe=recipe, stop_after_step=stop_after_step, checkpoint_every=2,
            validate_every=3, validation_repeats=1, visualized_images=1, log_every=1, decode_threads=2,
        ))

    run(tmp_path / "uninterrupted", None)
    run(tmp_path / "interrupted", 4)
    run(tmp_path / "interrupted", None)

    def losses(run_dir: Path) -> dict[int, float]:
        return {record["step"]: record["train/loss"] for record in _metrics(run_dir) if "train/loss" in record}

    assert losses(tmp_path / "interrupted") == losses(tmp_path / "uninterrupted")
    assert sorted(losses(tmp_path / "uninterrupted")) == list(range(1, 8))
    validations = [record for record in _metrics(tmp_path / "uninterrupted") if "val/AP_t1" in record]
    assert [record["step"] for record in validations] == [3, 6, 7]
    assert (tmp_path / "uninterrupted" / "images" / "step_0000003").is_dir()

    def detector(run_dir: Path) -> dict[str, np.ndarray]:
        model = CanViTForObjectDetection.from_pretrained(run_dir / "checkpoints" / "step_0000007" / "model")
        flat = nnx.to_flat_state(nnx.state(model))
        return {".".join(map(str, path)): np.asarray(leaf[...]) for path, leaf in zip(flat.paths, flat.leaves)}

    expected, actual = detector(tmp_path / "uninterrupted"), detector(tmp_path / "interrupted")
    for name, value in expected.items():
        np.testing.assert_array_equal(actual[name], value, err_msg=name)
    assert dataclasses.asdict(recipe) == json.loads((tmp_path / "interrupted" / "recipe.json").read_text())
