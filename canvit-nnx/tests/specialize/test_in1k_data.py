import io
import itertools
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from tfrecord.writer import TFRecordWriter

from canvit_nnx.specialize.in1k.data import PADDING_LABEL, training_batches, validation_batches

SCENE_SIZE_PX = 32


def _write_shard(path: Path, labels: list[int]) -> None:
    """A TFRecord shard of noise JPEGs of varied sizes, with ImageNet's fields and 1-based labels."""
    rng = np.random.default_rng(len(labels))
    writer = TFRecordWriter(str(path))
    for label in labels:
        height, width = (int(size) for size in rng.integers(20, 60, 2))
        buffer = io.BytesIO()
        Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8)).save(buffer, format="JPEG")
        writer.write({"image/encoded": (buffer.getvalue(), "byte"), "image/class/label": (label, "int")})
    writer.close()


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    _write_shard(tmp_path / "train-00000-of-00002", [1, 2, 3])
    _write_shard(tmp_path / "train-00001-of-00002", [1000, 999])
    _write_shard(tmp_path / "validation-00000-of-00001", [5, 6, 7, 8, 9])
    return tmp_path


def _training_batches(data_dir: Path, seed: tuple[int, ...], *, threads: int, host: int = 0, num_hosts: int = 1) -> list:
    batches = training_batches(
        data_dir, batch_size=4, scene_size_px=SCENE_SIZE_PX, seed=seed, host=host, num_hosts=num_hosts, threads=threads
    )
    return list(itertools.islice(batches, 3))


def test_training_batches_depend_only_on_the_seed(data_dir: Path) -> None:
    def batches(seed: tuple[int, ...], threads: int) -> list:
        return _training_batches(data_dir, seed, threads=threads)

    first, again, other = batches((0, 0), 4), batches((0, 0), 1), batches((0, 1), 4)
    assert first[0].scenes.shape == (4, SCENE_SIZE_PX, SCENE_SIZE_PX, 3) and first[0].scenes.dtype == np.float32
    for batch, repeated in zip(first, again, strict=True):
        np.testing.assert_array_equal(batch.scenes, repeated.scenes)
        np.testing.assert_array_equal(batch.labels, repeated.labels)
    assert not all(np.array_equal(batch.scenes, changed.scenes) for batch, changed in zip(first, other, strict=True))
    # Three batches of four read the five records more than twice, as class indices from 0.
    labels = np.concatenate([batch.labels for batch in first])
    assert sorted(set(labels.tolist())) == [0, 1, 2, 998, 999]


def test_hosts_read_disjoint_training_shards(data_dir: Path) -> None:
    labels = [
        {int(label) for batch in _training_batches(data_dir, (0, 0), threads=2, host=host, num_hosts=2)
         for label in batch.labels}
        for host in range(2)
    ]
    assert labels == [{0, 1, 2}, {998, 999}]


def _validation(data_dir: Path, *, num_images: int, batch_size: int, rows: slice) -> list:
    return list(validation_batches(
        data_dir, num_images=num_images, batch_size=batch_size, rows=rows, scene_size_px=SCENE_SIZE_PX, threads=3
    ))


def test_validation_reads_every_record_once_and_pads_the_last_batch(data_dir: Path) -> None:
    batches = _validation(data_dir, num_images=5, batch_size=2, rows=slice(0, 2))
    assert [batch.labels.tolist() for batch in batches] == [[4, 5], [6, 7], [8, PADDING_LABEL]]
    np.testing.assert_array_equal(batches[-1].scenes[1], 0.0)
    assert np.all(batches[-1].scenes[0] != 0.0)


def test_hosts_share_each_validation_batch(data_dir: Path) -> None:
    whole = _validation(data_dir, num_images=5, batch_size=4, rows=slice(0, 4))
    shares = [_validation(data_dir, num_images=5, batch_size=4, rows=slice(start, start + 2)) for start in (0, 2)]
    assert [len(share) for share in shares] == [2, 2], "every host steps through every batch"
    for index, batch in enumerate(whole):
        np.testing.assert_array_equal(np.concatenate([share[index].labels for share in shares]), batch.labels)
        np.testing.assert_array_equal(np.concatenate([share[index].scenes for share in shares]), batch.scenes)


def test_a_validation_set_of_another_size_fails(data_dir: Path) -> None:
    with pytest.raises(ValueError, match="more than 4"):
        _validation(data_dir, num_images=4, batch_size=2, rows=slice(0, 2))
    with pytest.raises(ValueError, match="end before 6"):
        _validation(data_dir, num_images=6, batch_size=2, rows=slice(0, 2))


def test_a_label_outside_imagenet_fails_with_its_shard(tmp_path: Path) -> None:
    _write_shard(tmp_path / "validation-00000-of-00001", [1001])
    with pytest.raises(ValueError, match="validation-00000-of-00001 record 0: label 1001"):
        next(validation_batches(
            tmp_path, num_images=1, batch_size=1, rows=slice(0, 1), scene_size_px=SCENE_SIZE_PX, threads=1
        ))
