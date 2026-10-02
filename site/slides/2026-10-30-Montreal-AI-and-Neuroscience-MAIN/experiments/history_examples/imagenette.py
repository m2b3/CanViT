"""Imagenette 2's "val" photos (ImageNet-1k photos of ten classes, one folder per WordNet id) and their ImageNet-1k
classes. Most are ImageNet-1k training photos (<wnid>_<n>.JPEG), which the DINOv3 probe was trained on; a few are
ImageNet-1k validation photos (ILSVRC2012_val_<n>.JPEG), held out from it."""

import json
from collections import Counter
from pathlib import Path

from canvit_pytorch.benchmarks.imagenet import CLASS_NAMES

IMAGENET_VAL_PREFIX = "ILSVRC2012_val_"

# WordNet id -> (ImageNet-1k class index, its name): the index is the WordNet id's rank among ImageNet-1k's 1,000
# sorted ids (timm's imagenet_synsets.txt); class_indices checks it against the names and ImageNet-ReAL.
CLASSES = {
    "n01440764": (0, "tench"),
    "n02102040": (217, "English springer"),
    "n02979186": (482, "cassette player"),
    "n03000684": (491, "chain saw"),
    "n03028079": (497, "church"),
    "n03394916": (566, "French horn"),
    "n03417042": (569, "garbage truck"),
    "n03425413": (571, "gas pump"),
    "n03445777": (574, "golf ball"),
    "n03888257": (701, "parachute"),
}


def imagenet_split(filename: str) -> str:
    """The ImageNet-1k split a photo comes from, by its file name."""
    stem = Path(filename).stem
    if stem.startswith(IMAGENET_VAL_PREFIX):
        return "val"
    assert stem.startswith("n") and "_" in stem, filename
    return "train"


def val_number(filename: str) -> int:
    """'ILSVRC2012_val_00009111.JPEG' -> 9111."""
    assert imagenet_split(filename) == "val", filename
    return int(Path(filename).stem.removeprefix(IMAGENET_VAL_PREFIX))


def class_indices(imagenette_val: Path, real_labels: Path) -> dict[str, int]:
    """WordNet id -> ImageNet-1k class index of each Imagenette folder, from CLASSES, checked: the folders are
    CLASSES's ids, each index has its name, and it is the most common ImageNet-ReAL label of the folder's validation
    photos. real_labels: ImageNet-ReAL's real.json, index i -> valid classes of ILSVRC2012_val_{i+1}."""
    folders = sorted(folder.name for folder in imagenette_val.iterdir() if folder.is_dir())
    assert folders == sorted(CLASSES), f"{imagenette_val}: folders {folders}"
    labels: list[list[int]] = json.loads(real_labels.read_text())
    for wnid, (index, name) in CLASSES.items():
        assert CLASS_NAMES[index] == name, (wnid, index, CLASS_NAMES[index], name)
        val_photos = [photo.name for photo in (imagenette_val / wnid).iterdir() if imagenet_split(photo.name) == "val"]
        votes = Counter(label for photo in val_photos for label in labels[val_number(photo) - 1]).most_common(2)
        assert votes[0][0] == index and (len(votes) == 1 or votes[0][1] > votes[1][1]), (wnid, index, votes)
    return {wnid: index for wnid, (index, _) in CLASSES.items()}
