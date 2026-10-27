"""ImageNet-1k's validation images as GFNet's and AdaptiveNN's evaluators read them."""

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from torchvision.datasets import ImageFolder

NUM_CLASSES = 1000


@contextmanager
def evaluation_root(validation_dir: Path) -> Iterator[Path]:
    """A directory whose `train` and `val` both link to the validation images, in ImageFolder layout.

    Both evaluators build a training set even when they only evaluate, and
    neither reads it then; the link spares them listing the training images.
    """
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for split in ("train", "val"):
            (root / split).symlink_to(validation_dir.resolve(), target_is_directory=True)
        yield root


def num_images(validation_dir: Path) -> int:
    dataset = ImageFolder(str(validation_dir))
    assert len(dataset.classes) == NUM_CLASSES, f"{validation_dir}: {len(dataset.classes)} classes"
    return len(dataset)
