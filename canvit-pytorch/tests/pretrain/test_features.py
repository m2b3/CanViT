"""The feature tools chain: index a class-per-directory tree, export shards, check them, read training batches."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from PIL import Image

from canvit_pytorch.pretrain.data import TrainingBatches
from canvit_pytorch.pretrain.features import check, export, index
from canvit_pytorch.pretrain.features.shard import load_shard
from canvit_pytorch.teacher import TEACHER_REPO

SCENE_PX = 64
CLASSES, IMAGES_PER_CLASS, SHARD_SIZE = 2, 5, 4


@pytest.mark.slow
@pytest.mark.network
def test_index_export_check_read(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    image_root = tmp_path / "images"
    for c in range(CLASSES):
        (image_root / f"class{c}").mkdir(parents=True)
        for i in range(IMAGES_PER_CLASS):
            pixels = rng.integers(0, 256, (80, 96, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(image_root / f"class{c}" / f"{i}.jpg")
    n_images = CLASSES * IMAGES_PER_CLASS

    index_path = tmp_path / "index.parquet"
    index.main(index.Arguments(image_root=image_root, out=index_path))
    table = pq.read_table(index_path)
    assert sorted(table.column("path").to_pylist()) == sorted(str(p.relative_to(image_root)) for p in image_root.glob("*/*.jpg"))
    with pytest.raises(AssertionError):
        index.main(index.Arguments(image_root=image_root, out=index_path))

    out_dir = tmp_path / "features"
    export.main(export.Arguments(
        index=index_path, image_root=image_root, out_dir=out_dir, first_shard=0, end_shard=100,
        image_size_px=SCENE_PX, shard_size=SHARD_SIZE, batch_size=3, num_workers=0, device="cpu",
    ))
    check.main(check.Arguments(shards_dir=out_dir / "shards", expected_images=n_images))
    last = load_shard(out_dir / "shards" / "00002.pt")
    assert last["paths"] == table.column("path").to_pylist()[2 * SHARD_SIZE :]
    assert last["patches"].shape == (n_images - 2 * SHARD_SIZE, (SCENE_PX // 16) ** 2, 768)

    batches = TrainingBatches(
        shards_dir=out_dir / "shards", images_dir=image_root, scene_size_px=SCENE_PX, teacher_repo=TEACHER_REPO,
        batch_size=2, num_workers=0, start_step=1,
    )
    images, patches, cls = batches.next()
    assert images.shape == (2, 3, SCENE_PX, SCENE_PX) and patches.shape == (2, 16, 768) and cls.shape == (2, 768)
    first = load_shard(out_dir / "shards" / "00000.pt")
    assert (patches == first["patches"][2:4]).all(), "step 1 resumes at the second batch of the first shard"
