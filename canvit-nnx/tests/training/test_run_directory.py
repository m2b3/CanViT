from pathlib import Path

import optax
from canvit_core import CanViTConfig
from flax import nnx

from canvit_nnx import CanViTForImageClassification
from canvit_nnx.training.run_directory import MODEL, RunDirectory


def test_a_checkpoint_left_unnamed_by_a_stopped_save_is_saved_over(small_config: CanViTConfig, tmp_path: Path) -> None:
    # An invocation stopped between moving its checkpoint into place and naming it latest leaves the directory
    # behind; the resumed run reaches that step again from the previous checkpoint.
    model = CanViTForImageClassification(canvit_config=small_config, n_classes=3, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.sgd(0.1), wrt=nnx.Param)
    run_directory = RunDirectory(tmp_path, is_leader=True)
    run_directory.save_checkpoint(model=model, optimizer=optimizer)
    (run_directory.checkpoints / "step_0000001").mkdir()

    optimizer.step[...] = optimizer.step[...] + 1
    run_directory.save_checkpoint(model=model, optimizer=optimizer)

    assert run_directory.latest_checkpoint() == run_directory.checkpoints / "step_0000001"
    assert sorted(path.name for path in run_directory.checkpoints.glob("step_*")) == ["step_0000001"]
    CanViTForImageClassification.from_pretrained(run_directory.checkpoints / "step_0000001" / MODEL)
