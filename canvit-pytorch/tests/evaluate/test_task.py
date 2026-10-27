from pathlib import Path

import torch

from canvit_pytorch.evaluate.tasks.task import Task


def test_a_saved_result_loads_without_allowing_any_class(tmp_path: Path) -> None:
    task = Task(output=tmp_path / "result.pt", batch_size=1, device="cpu")
    path = task.save({"values": torch.arange(3)}, n_images=3)
    result = torch.load(path, weights_only=True)
    assert result["metadata"]["config"]["output"] == str(tmp_path / "result.pt")
    assert torch.equal(result["values"], torch.arange(3)) and result["metadata"]["n_images"] == 3
