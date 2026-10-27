import hashlib
from pathlib import Path

import pytest

from canvit_pytorch.hub.loading import checkpoint_record
from canvit_pytorch.probes import SegmentationProbe


def test_a_local_checkpoint_is_recorded_by_the_hash_of_the_weights_it_loads(tmp_path: Path) -> None:
    SegmentationProbe(embed_dim=8, num_classes=3, dropout=0.0, use_ln=True).save_pretrained(tmp_path)
    record = checkpoint_record(str(tmp_path))
    assert record == {"repo": str(tmp_path), "revision": None,
                      "weights_sha256": hashlib.sha256((tmp_path / "model.safetensors").read_bytes()).hexdigest()}
    with pytest.raises(AssertionError, match="has no revision"):
        checkpoint_record(str(tmp_path), revision="main")
