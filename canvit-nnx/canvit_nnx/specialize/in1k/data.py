"""ImageNet-1k scenes from local TFRecord shards, decoded by a thread pool into batches of normalized scenes.

The shards are the standard ImageNet TFRecords that canvit_pytorch.specialize.in1k_tpu
also reads: files named train-* and validation-* whose records hold the fields
image/encoded (JPEG) and image/class/label (1 to 1000).
"""

import io
import itertools
import math
from collections.abc import Generator, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
from canvit_core.preprocess import imagenet_normalize, preprocess
from PIL import Image
from tfrecord.reader import tfrecord_loader

from canvit_nnx.training.host_data import host_batches, map_in_order

NUM_CLASSES = 1000
NUM_TRAIN_IMAGES = 1_281_167
NUM_VALIDATION_IMAGES = 50_000
PADDING_LABEL = -1
"""Label of the zero scenes that fill the last validation batch."""

TFRECORD_FIELDS = {"image/encoded": "byte", "image/class/label": "int"}
CROP_SCALE = (0.2, 1.0)
CROP_ASPECT_RATIO = (3 / 4, 4 / 3)

Split = Literal["train", "validation"]


class SceneBatch(NamedTuple):
    scenes: np.ndarray  # float32 [B, S, S, 3], ImageNet-normalized
    labels: np.ndarray  # int32 [B], class indices from 0, or PADDING_LABEL


class Record(NamedTuple):
    jpeg: bytes
    label: int  # class index from 0
    source: str  # shard and record index, for errors


def shards(data_dir: Path, split: Split) -> list[Path]:
    """The split's shard files, without index files (which have a suffix)."""
    found = sorted(path for path in data_dir.glob(f"{split}-*") if not path.suffix)
    assert found, f"no {split}-* TFRecord shards in {data_dir}"
    return found


def records(shard_paths: Iterable[Path]) -> Iterator[Record]:
    for shard in shard_paths:
        for index, record in enumerate(tfrecord_loader(str(shard), None, TFRECORD_FIELDS)):
            assert isinstance(record, dict), type(record)
            label = int(record["image/class/label"].item())
            if not 1 <= label <= NUM_CLASSES:
                raise ValueError(f"{shard} record {index}: label {label} is outside 1..{NUM_CLASSES}")
            jpeg = record["image/encoded"]
            assert isinstance(jpeg, bytes), type(jpeg)
            yield Record(jpeg=jpeg, label=label - 1, source=f"{shard} record {index}")


def _decode(record: Record) -> Image.Image:
    try:
        return Image.open(io.BytesIO(record.jpeg)).convert("RGB")
    except OSError as error:
        raise ValueError(f"{record.source}: cannot decode the image") from error


def random_resized_crop_box(width: int, height: int, rng: np.random.Generator) -> tuple[int, int, int, int]:
    """(left, top, right, bottom) of torchvision's RandomResizedCrop with CROP_SCALE and CROP_ASPECT_RATIO."""
    area = width * height
    log_ratios = np.log(CROP_ASPECT_RATIO)
    for _ in range(10):
        target_area = area * rng.uniform(*CROP_SCALE)
        aspect_ratio = math.exp(rng.uniform(*log_ratios))
        crop_width = round(math.sqrt(target_area * aspect_ratio))
        crop_height = round(math.sqrt(target_area / aspect_ratio))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            top = int(rng.integers(0, height - crop_height + 1))
            left = int(rng.integers(0, width - crop_width + 1))
            return left, top, left + crop_width, top + crop_height
    ratio = width / height
    if ratio < CROP_ASPECT_RATIO[0]:
        crop_width, crop_height = width, round(width / CROP_ASPECT_RATIO[0])
    elif ratio > CROP_ASPECT_RATIO[1]:
        crop_width, crop_height = round(height * CROP_ASPECT_RATIO[1]), height
    else:
        crop_width, crop_height = width, height
    top, left = (height - crop_height) // 2, (width - crop_width) // 2
    return left, top, left + crop_width, top + crop_height


def training_scene(image: Image.Image, rng: np.random.Generator, *, size: int) -> np.ndarray:
    """A random resized crop, flipped horizontally half the time, as the PyTorch trainer's transform."""
    scene = image.crop(random_resized_crop_box(image.width, image.height, rng)).resize(
        (size, size), Image.Resampling.BILINEAR
    )
    if rng.random() < 0.5:
        scene = scene.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return imagenet_normalize(np.asarray(scene, dtype=np.float32) / np.float32(255.0))


def training_batches(
    data_dir: Path, *, batch_size: int, scene_size_px: int, seed: Sequence[int], host: int, num_hosts: int, threads: int
) -> Generator[SceneBatch]:
    """Batches of batch_size scenes from endless passes over this host's training shards.

    Each of num_hosts hosts reads every num_hosts-th shard. Each pass reads the
    shards in a new random order and each shard's records in file order. A
    scene's crop and flip depend only on seed, the host and the scene's
    position in the host's stream, whatever the thread scheduling.
    """
    shard_paths = shards(data_dir, "train")[host::num_hosts]
    assert shard_paths, f"host {host} of {num_hosts} has no training shard in {data_dir}"

    def stream() -> Iterator[tuple[Record, np.random.Generator]]:
        for pass_index in itertools.count():
            order = np.random.default_rng([*seed, pass_index]).permutation(len(shard_paths))
            for position, record in enumerate(records(shard_paths[index] for index in order)):
                yield record, np.random.default_rng([*seed, pass_index, position * num_hosts + host])

    def example(item: tuple[Record, np.random.Generator]) -> tuple[np.ndarray, int]:
        record, rng = item
        return training_scene(_decode(record), rng, size=scene_size_px), record.label

    examples = map_in_order(example, stream(), threads=threads)
    while True:
        scenes, labels = zip(*itertools.islice(examples, batch_size), strict=True)
        yield SceneBatch(scenes=np.stack(scenes), labels=np.array(labels, dtype=np.int32))


def validation_batches(
    data_dir: Path, *, num_images: int, batch_size: int, rows: slice, scene_size_px: int, threads: int
) -> Generator[SceneBatch]:
    """This host's rows of the global batches of the num_images validation images, in order.

    Each scene is resized and center-cropped as in evaluation; zero scenes
    labeled PADDING_LABEL fill the last batch.
    """
    transform = preprocess(scene_size_px)

    def example(record: Record) -> tuple[np.ndarray, int]:
        return transform(_decode(record)), record.label

    for examples in host_batches(
        records(shards(data_dir, "validation")), example, num_items=num_images, batch_size=batch_size,
        rows=rows, threads=threads,
    ):
        scenes = np.zeros((len(examples), scene_size_px, scene_size_px, 3), dtype=np.float32)
        labels = np.full((len(examples),), PADDING_LABEL, dtype=np.int32)
        for row, example_or_padding in enumerate(examples):
            if example_or_padding is not None:
                scenes[row], labels[row] = example_or_padding
        yield SceneBatch(scenes=scenes, labels=labels)
