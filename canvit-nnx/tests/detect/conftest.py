import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# COCO's 80 categories have ids from 1 to 90 with gaps; these do too, listed out of order.
CATEGORY_IDS = [90, *range(1, 80)]


def _write_split(data_dir: Path, split: str, *, num_images: int, seed: int) -> None:
    """Noise JPEGs of varied sizes, each with objects of random classes, one crowd region and one empty box."""
    rng = np.random.default_rng(seed)
    (data_dir / split).mkdir(parents=True)
    images, annotations = [], []
    for index in range(num_images):
        image_id = int(1000 - 7 * index)  # listed in decreasing id order
        height, width = (int(size) for size in rng.integers(30, 70, 2))
        Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8)).save(
            data_dir / split / f"{image_id:012d}.jpg"
        )
        images.append({"id": image_id, "file_name": f"{image_id:012d}.jpg", "height": height, "width": width})
        for kind in ["object"] * int(rng.integers(0, 4)) + ["crowd", "empty"]:
            x, y = float(rng.uniform(0, width / 2)), float(rng.uniform(0, height / 2))
            box_width, box_height = float(rng.uniform(2, width / 2)), float(rng.uniform(2, height / 2))
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": CATEGORY_IDS[int(rng.integers(len(CATEGORY_IDS)))],
                "bbox": [x, y, 0.0 if kind == "empty" else box_width, box_height],
                "area": box_width * box_height,
                "iscrowd": int(kind == "crowd"),
            })
    categories = [{"id": category_id, "name": f"class {category_id}"} for category_id in CATEGORY_IDS]
    (data_dir / "annotations").mkdir(exist_ok=True)
    (data_dir / "annotations" / f"instances_{split}.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": categories})
    )


@pytest.fixture(scope="module")
def coco_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    data_dir = tmp_path_factory.mktemp("coco")
    _write_split(data_dir, "train2017", num_images=7, seed=0)
    _write_split(data_dir, "val2017", num_images=5, seed=1)
    return data_dir
