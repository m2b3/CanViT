import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from canvit_nnx.detect.data import (
    PADDING_IMAGE_ID,
    annotation_file,
    read_split,
    training_batches,
    validation_batches,
)
from canvit_nnx.detect.decode import Detections
from canvit_nnx.detect.scoring import coco_detections, coco_statistics, load_ground_truth

SCENE_SIZE_PX, GRID_SIZE = 32, 4


def test_reading_keeps_objects_in_annotation_order_without_crowds_or_empty_boxes(coco_dir: Path) -> None:
    split = read_split(coco_dir, "val2017")
    record = json.loads(annotation_file(coco_dir, "val2017").read_text())
    assert [image.image_id for image in split.images] == sorted(image["id"] for image in record["images"])
    assert split.category_ids == tuple(sorted(split.category_ids))
    for image in split.images:
        objects = [
            annotation for annotation in record["annotations"]
            if annotation["image_id"] == image.image_id and not annotation["iscrowd"] and annotation["bbox"][2] > 0
        ]
        assert [split.category_ids[label] for label in image.labels] == [obj["category_id"] for obj in objects]
        for box, obj in zip(image.boxes, objects, strict=True):
            x, y, width, height = obj["bbox"]
            np.testing.assert_allclose(
                box, [y / image.height, x / image.width, (y + height) / image.height, (x + width) / image.width],
                rtol=1e-6,
            )


def _training(coco_dir: Path, *, first_step: int, rows: slice, num_batches: int) -> list:
    batches = training_batches(
        read_split(coco_dir, "train2017").images,
        batch_size=4,
        rows=rows,
        first_step=first_step,
        scene_size_px=SCENE_SIZE_PX,
        grid_size=GRID_SIZE,
        seed=0,
        threads=3,
    )
    return list(itertools.islice(batches, num_batches))


def test_a_resumed_run_sees_the_batches_of_an_uninterrupted_run(coco_dir: Path) -> None:
    uninterrupted = _training(coco_dir, first_step=0, rows=slice(0, 4), num_batches=5)
    resumed = _training(coco_dir, first_step=3, rows=slice(0, 4), num_batches=2)
    for expected, actual in zip(uninterrupted[3:], resumed, strict=True):
        np.testing.assert_array_equal(actual.scenes, expected.scenes)
        for name, field in expected.targets._asdict().items():
            np.testing.assert_array_equal(getattr(actual.targets, name), field, err_msg=name)


def test_hosts_load_their_rows_of_each_batch(coco_dir: Path) -> None:
    whole = _training(coco_dir, first_step=1, rows=slice(0, 4), num_batches=2)
    shares = [_training(coco_dir, first_step=1, rows=slice(start, start + 2), num_batches=2) for start in (0, 2)]
    for index, batch in enumerate(whole):
        np.testing.assert_array_equal(np.concatenate([share[index].scenes for share in shares]), batch.scenes)
        np.testing.assert_array_equal(
            np.concatenate([share[index].targets.heatmap for share in shares]), batch.targets.heatmap
        )


def test_validation_shares_cover_every_image_once(coco_dir: Path) -> None:
    images = read_split(coco_dir, "val2017").images
    shares = [
        list(validation_batches(images, batch_size=4, rows=slice(start, start + 2), scene_size_px=SCENE_SIZE_PX, threads=2))
        for start in (0, 2)
    ]
    ids = np.concatenate([np.concatenate([share[index].image_ids for share in shares]) for index in range(2)])
    assert ids.tolist() == [image.image_id for image in images] + [PADDING_IMAGE_ID] * 3


def test_true_boxes_score_an_average_precision_of_one(coco_dir: Path, tmp_path: Path) -> None:
    split = read_split(coco_dir, "val2017")
    # Ground truth without its empty boxes, which no detection can match.
    record = json.loads(annotation_file(coco_dir, "val2017").read_text())
    record["annotations"] = [annotation for annotation in record["annotations"] if annotation["bbox"][2] > 0]
    (tmp_path / "ground_truth.json").write_text(json.dumps(record))
    images = [image for image in split.images if len(image.labels)]
    num_detections = max(len(image.labels) for image in images)

    def padded(values: np.ndarray, fill: float) -> np.ndarray:
        return np.concatenate([values, np.full((num_detections - len(values), *values.shape[1:]), fill)])

    detections = Detections(
        boxes=np.stack([padded(image.boxes, 0.0) for image in images]),
        scores=np.stack([padded(np.ones(len(image.labels)), 0.0) for image in images]),
        labels=np.stack([padded(image.labels, 0).astype(np.int32) for image in images]),
    )
    rows = coco_detections(
        detections,
        image_ids=np.array([image.image_id for image in images]),
        heights=np.array([image.height for image in images]),
        widths=np.array([image.width for image in images]),
        category_ids=split.category_ids,
    )
    statistics = coco_statistics(
        rows,
        ground_truth=load_ground_truth(tmp_path / "ground_truth.json"),
        image_ids=np.array([image.image_id for image in images]),
    )
    assert statistics["AP"] == pytest.approx(1.0) and statistics["AR_max100"] == pytest.approx(1.0)
