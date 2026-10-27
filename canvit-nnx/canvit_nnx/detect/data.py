"""COCO 2017 detection scenes and their CenterNet targets, decoded on a thread pool.

The data directory has COCO's own layout: annotations/instances_<split>.json
and the <split>/ image directory. An image becomes a square scene by resizing
each axis independently, which leaves normalized box coordinates unchanged.
Scenes stay uint8 on the host and in transfer, a quarter of the bytes of
float32, and are normalized on the device.
"""

import itertools
import json
from collections import defaultdict
from collections.abc import Generator, Iterator, Sequence
from pathlib import Path
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from canvit_nnx.detect.photometric import photometric_distortion
from canvit_nnx.detect.targets import Targets, render
from canvit_nnx.preprocess import imagenet_normalize
from canvit_nnx.training.host_data import host_batches, host_items, map_in_order

NUM_CLASSES = 80
NUM_TRAINING_IMAGES = 118_287
"""Images of train2017, those without objects included."""
MAX_OBJECTS = 100
"""Objects kept per image, in annotation order; COCO's most crowded image has 93."""
PADDING_IMAGE_ID = -1
"""Image id of the zero scenes that fill the last validation batch."""

Split = Literal["train2017", "val2017"]


class CocoImage(NamedTuple):
    image_id: int
    path: Path
    height: int
    width: int
    boxes: np.ndarray  # float32 [N, 4] normalized (ymin, xmin, ymax, xmax), clipped to [0, 1]
    labels: np.ndarray  # int32 [N] class indices


class CocoSplit(NamedTuple):
    images: list[CocoImage]  # in increasing id order
    category_ids: tuple[int, ...]  # COCO's category id of each class index, in increasing order
    category_names: tuple[str, ...]


class TrainingBatch(NamedTuple):
    scenes: np.ndarray  # uint8 [B, S, S, 3]
    targets: Targets[np.ndarray]


class ValidationBatch(NamedTuple):
    scenes: np.ndarray  # uint8 [B, S, S, 3]
    image_ids: np.ndarray  # int32 [B], PADDING_IMAGE_ID on padding


def annotation_file(data_dir: Path, split: Split) -> Path:
    return data_dir / "annotations" / f"instances_{split}.json"


def read_split(data_dir: Path, split: Split) -> CocoSplit:
    """The split's images with their objects; crowd regions and empty boxes are left out."""
    record = json.loads(annotation_file(data_dir, split).read_text())
    categories = sorted(record["categories"], key=lambda category: category["id"])
    assert len(categories) == NUM_CLASSES, len(categories)
    label_of = {category["id"]: label for label, category in enumerate(categories)}
    annotations_of = defaultdict(list)
    for annotation in record["annotations"]:
        annotations_of[annotation["image_id"]].append(annotation)

    images = []
    for info in sorted(record["images"], key=lambda info: info["id"]):
        height, width = info["height"], info["width"]
        boxes, labels = [], []
        for annotation in annotations_of[info["id"]]:
            if annotation.get("iscrowd", 0):
                continue
            if len(boxes) == MAX_OBJECTS:
                break
            x, y, box_width, box_height = annotation["bbox"]
            if box_width <= 0 or box_height <= 0:
                continue
            boxes.append((y / height, x / width, (y + box_height) / height, (x + box_width) / width))
            labels.append(label_of[annotation["category_id"]])
        images.append(
            CocoImage(
                image_id=info["id"],
                path=data_dir / split / info["file_name"],
                height=height,
                width=width,
                boxes=np.clip(np.array(boxes, dtype=np.float32).reshape(-1, 4), 0.0, 1.0),
                labels=np.array(labels, dtype=np.int32),
            )
        )
    return CocoSplit(
        images=images,
        category_ids=tuple(category["id"] for category in categories),
        category_names=tuple(category["name"] for category in categories),
    )


def scene(image: CocoImage, *, size: int) -> np.ndarray:
    """The image resized to a [size, size, 3] uint8 scene."""
    try:
        with Image.open(image.path) as file:
            return np.asarray(file.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)
    except OSError as error:
        raise ValueError(f"image {image.image_id}: cannot decode {image.path}") from error


def training_example(
    image: CocoImage, rng: np.random.Generator, *, scene_size_px: int, grid_size: int
) -> tuple[np.ndarray, Targets[np.ndarray]]:
    """A scene flipped horizontally half the time and photometrically distorted, with its targets."""
    pixels, boxes = scene(image, size=scene_size_px), image.boxes
    if rng.random() < 0.5:
        pixels = pixels[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 1], boxes[:, 3] = 1.0 - image.boxes[:, 3], 1.0 - image.boxes[:, 1]
    targets = render(boxes, image.labels, grid_size=grid_size, num_classes=NUM_CLASSES)
    return photometric_distortion(pixels, rng), targets


def training_batches(
    images: Sequence[CocoImage],
    *,
    batch_size: int,
    rows: slice,
    first_step: int,
    scene_size_px: int,
    grid_size: int,
    seed: int,
    threads: int,
) -> Generator[TrainingBatch]:
    """This host's rows of the global batches from first_step on, in endless passes over the images.

    Each pass visits the images in a new random order. A scene, its flip and
    its distortion depend only on seed and on its position in the global
    stream, so a run resumed at any step sees what an uninterrupted run would.
    """

    def stream() -> Iterator[tuple[CocoImage, np.random.Generator]]:
        order_pass, order = -1, np.empty(0, dtype=np.int64)
        for position in itertools.count(first_step * batch_size):
            pass_index, offset = divmod(position, len(images))
            if pass_index != order_pass:
                order_pass, order = pass_index, np.random.default_rng([seed, pass_index]).permutation(len(images))
            yield images[order[offset]], np.random.default_rng([seed, pass_index, offset])

    def example(item: tuple[CocoImage, np.random.Generator]) -> tuple[np.ndarray, Targets[np.ndarray]]:
        return training_example(*item, scene_size_px=scene_size_px, grid_size=grid_size)

    examples = map_in_order(example, host_items(stream(), batch_size=batch_size, rows=rows), threads=threads)
    while True:
        scenes, targets = zip(*itertools.islice(examples, rows.stop - rows.start), strict=True)
        yield TrainingBatch(scenes=np.stack(scenes), targets=Targets(*map(np.stack, zip(*targets, strict=True))))


def validation_batches(
    images: Sequence[CocoImage], *, batch_size: int, rows: slice, scene_size_px: int, threads: int
) -> Generator[ValidationBatch]:
    """This host's rows of the images' global batches, in order; zero scenes fill the last batch."""
    for chunk in host_batches(
        images, lambda image: (image.image_id, scene(image, size=scene_size_px)),
        num_items=len(images), batch_size=batch_size, rows=rows, threads=threads,
    ):
        batch = ValidationBatch(
            scenes=np.zeros((len(chunk), scene_size_px, scene_size_px, 3), dtype=np.uint8),
            image_ids=np.full((len(chunk),), PADDING_IMAGE_ID, dtype=np.int32),
        )
        for row, image_or_padding in enumerate(chunk):
            if image_or_padding is not None:
                batch.image_ids[row], batch.scenes[row] = image_or_padding
        yield batch


def normalize(scenes: jax.Array) -> jax.Array:
    """ImageNet-normalize uint8 scenes into float32."""
    return imagenet_normalize(scenes.astype(jnp.float32) / 255.0)
