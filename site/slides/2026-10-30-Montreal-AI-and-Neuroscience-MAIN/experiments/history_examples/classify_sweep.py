"""DINOv3 ViT-B/16 and its linear ImageNet-1k probe on every Imagenette validation photo, for choosing #history's
classification example. The probe reads the CLS token after DINOv3's final LayerNorm of a 512 px scene, as in
canvit_pytorch.pretrain.monitor.validation (teacher(images).cls -> load_teacher_classifier()). Two preprocessings, both
short side to 512 then center crop: canvit_pytorch.preprocess.preprocess (bilinear, how CanViT's code feeds this probe)
and the probe repository's own val transform (bicubic, dinov3_in1k_probes.data.make_val_transform), to see whether the
probabilities depend on it. Writes a table, one row per photo, unranked, with the ImageNet-1k split it comes from."""

import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from canvit_pytorch.benchmarks.imagenet import CLASS_NAMES
from canvit_pytorch.hub.repos import RELEASED_SCENE_SIZE_PX
from canvit_pytorch.preprocess import IMAGENET_MEAN, IMAGENET_STD, preprocess
from canvit_pytorch.pretrain.monitor.validation import TEACHER_CLASSIFIER_SIZE_PX, load_teacher_classifier
from canvit_pytorch.teacher import TEACHER_REPO, load_teacher
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from experiments import logs
from experiments.history_examples.imagenette import class_indices, imagenet_split
from experiments.outputs import WORK

log = logging.getLogger(__name__)

BATCH_SIZE = 16
NUM_WORKERS = 6
CPU_CHECK_TOLERANCE = 1e-3  # max |p_device - p_cpu| over the first batch's softmax


@dataclass(frozen=True)
class Config:
    imagenette: Path
    """Imagenette 2's val/ directory (imagenette2/val)"""
    real_labels: Path
    """ImageNet-ReAL's real.json (github.com/m2b3/dinov3-in1k-probes, dinov3_in1k_probes/data/in1k/real.json)"""
    out: Path = WORK / "history/classification/scores.tsv"
    device: str = "mps"


def probe_repo_transform(size_px: int) -> transforms.Compose:
    """dinov3_in1k_probes.data.make_val_transform at github.com/m2b3/dinov3-in1k-probes @ 20114de."""
    return transforms.Compose([
        transforms.Resize(size_px, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size_px),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class Photos(Dataset[tuple[Tensor, Tensor, int]]):
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        self.canvit_transform = preprocess(RELEASED_SCENE_SIZE_PX)
        self.probe_repo_transform = probe_repo_transform(RELEASED_SCENE_SIZE_PX)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, int]:
        image = Image.open(self.paths[index]).convert("RGB")
        return self.canvit_transform(image), self.probe_repo_transform(image), index


@torch.inference_mode()
def probabilities(teacher: torch.nn.Module, classifier: torch.nn.Module, images: Tensor) -> Tensor:
    return torch.softmax(classifier(teacher(images).cls.float()), dim=-1)


def main(cfg: Config) -> None:
    assert RELEASED_SCENE_SIZE_PX == TEACHER_CLASSIFIER_SIZE_PX
    indices = class_indices(cfg.imagenette, cfg.real_labels)
    for wnid, index in indices.items():
        log.info("%s -> ImageNet-1k class %d %r", wnid, index, CLASS_NAMES[index])
    paths = sorted(cfg.imagenette.glob("*/*.JPEG"))
    log.info("%d photos in %s", len(paths), cfg.imagenette)
    device = torch.device(cfg.device)
    teacher = load_teacher(TEACHER_REPO, device)
    classifier = load_teacher_classifier(device)
    assert classifier.in_features == teacher.embed_dim and classifier.out_features == len(CLASS_NAMES)

    loader = DataLoader(Photos(paths), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "image", "imagenet_split", "wnid", "class_index", "class_name", "width", "height", "short_side",
        "prob_true", "top1_index", "top1_name", "top1_prob", "top2_name", "top2_prob",
        "prob_true_bicubic", "top1_index_bicubic",
    ]
    num_correct, start = 0, time.monotonic()
    with cfg.out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for batch_number, (canvit_images, bicubic_images, batch_indices) in enumerate(loader):
            probs = probabilities(teacher, classifier, canvit_images.to(device)).cpu()
            probs_bicubic = probabilities(teacher, classifier, bicubic_images.to(device)).cpu()
            if batch_number == 0:
                teacher.cpu()
                classifier.cpu()
                probs_cpu = probabilities(teacher, classifier, canvit_images)
                teacher.to(device)
                classifier.to(device)
                difference = (probs - probs_cpu).abs().max().item()
                log.info("first batch, %s against CPU: max |dp| = %.2e, same argmax: %s", cfg.device,
                         difference, bool((probs.argmax(-1) == probs_cpu.argmax(-1)).all()))
                assert difference < CPU_CHECK_TOLERANCE, difference
            top2 = probs.topk(2, dim=-1)
            for row, index in enumerate(batch_indices.tolist()):
                path = paths[index]
                width, height = Image.open(path).size
                true_index = indices[path.parent.name]
                top1_index = int(top2.indices[row, 0])
                num_correct += top1_index == true_index
                writer.writerow({
                    "image": f"{path.parent.name}/{path.name}", "imagenet_split": imagenet_split(path.name),
                    "wnid": path.parent.name,
                    "class_index": true_index, "class_name": CLASS_NAMES[true_index],
                    "width": width, "height": height, "short_side": min(width, height),
                    "prob_true": f"{probs[row, true_index].item():.6f}",
                    "top1_index": top1_index, "top1_name": CLASS_NAMES[top1_index],
                    "top1_prob": f"{top2.values[row, 0].item():.6f}",
                    "top2_name": CLASS_NAMES[int(top2.indices[row, 1])], "top2_prob": f"{top2.values[row, 1].item():.6f}",
                    "prob_true_bicubic": f"{probs_bicubic[row, true_index].item():.6f}",
                    "top1_index_bicubic": int(probs_bicubic[row].argmax()),
                })
            if batch_number % 20 == 0:
                log.info("%d/%d photos, %.1f s", min((batch_number + 1) * BATCH_SIZE, len(paths)), len(paths),
                         time.monotonic() - start)
    log.info("top-1 accuracy over %d Imagenette val photos: %.2f%% -> %s", len(paths), 100 * num_correct / len(paths), cfg.out)


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
