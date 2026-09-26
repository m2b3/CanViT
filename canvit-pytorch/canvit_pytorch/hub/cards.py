"""Hub model cards (README.md) of CanViT checkpoints and probes.

Every card shares the project's description, links and citation (project.py)
and shows working 0.2 code for its checkpoint; the facts that differ between
checkpoints come in as typed arguments.
"""

from dataclasses import dataclass

from huggingface_hub import EvalResult, ModelCardData

from canvit_pytorch import project
from canvit_pytorch.hub.loading import FORMAT_VERSION
from canvit_pytorch.hub.repos import PretrainingDataset
from canvit_pytorch.pretrain.ablations import Ablation

DATASET_NAMES: dict[PretrainingDataset, str] = {"in21k": "ImageNet-21k", "in1k": "ImageNet-1k"}
DATASET_HUB_IDS: dict[PretrainingDataset, str] = {"in21k": "imagenet-21k", "in1k": "imagenet-1k"}
ADE20K_HUB_ID = "scene_parse_150"
TAGS = ["canvit", "active-vision", "vision-transformer"]


@dataclass(frozen=True)
class PretrainingFacts:
    dataset: PretrainingDataset
    teacher_repo: str
    scene_size_px: int
    glimpse_size_px: int
    canvas_grid_size: int
    steps: int | None


@dataclass(frozen=True)
class ProbeTraining:
    """How a segmentation probe was trained, following the paper's probing protocol."""

    steps: int
    batch_size: int
    peak_lr: float
    weight_decay: float
    warmup_steps: int
    dropout: float
    crop_scale_range: tuple[float, float]
    bfloat16: bool

    def rows(self) -> list[tuple[str, str]]:
        low, high = self.crop_scale_range
        return [
            ("Training steps", f"{self.steps:,}, batch size {self.batch_size}"),
            ("Optimizer", f"AdamW, peak learning rate {self.peak_lr:g}, weight decay {self.weight_decay:g}"),
            ("Schedule", f"{self.warmup_steps:,}-step linear warmup, then cosine decay"),
            ("Augmentation", f"random crops of scale {low:g} to {high:g}, horizontal flips"),
            ("Dropout", f"{self.dropout:g}"),
            ("Precision", "bfloat16 autocast" if self.bfloat16 else "float32"),
        ]


@dataclass(frozen=True)
class CanvasProbeFacts:
    pretrained_repo: str
    scene_size_px: int
    canvas_grid_size: int
    canvas_dim: int
    glimpse_size_px: int
    training_glimpses: int
    training_policy: str
    """The paper's name of the viewing policy during probe training."""
    training: ProbeTraining


@dataclass(frozen=True)
class DINOv3ProbeFacts:
    dinov3_repo: str
    dinov3_name: str
    """E.g. "DINOv3 ViT-B/16"."""
    input_size_px: int
    patch_size: int
    embed_dim: int
    training: ProbeTraining


def _card(*, data: ModelCardData, title: str, summary: str, usage: str, details: list[tuple[str, str]]) -> str:
    table = "\n".join(["| | |", "|---|---|", *(f"| {name} | {value} |" for name, value in details)])
    return f"""---
{data.to_yaml()}
---

# {title}

{summary}

{project.DESCRIPTION}

[Paper ({project.VENUE})]({project.PAPER_URL}) · [Code]({project.CODE_URL}) · [Project page]({project.PAGE_URL}) · [All checkpoints]({project.HUB_ORG_URL})

## Usage

```bash
pip install "canvit-pytorch>={FORMAT_VERSION}"
```

```python
{usage.strip()}
```

## Details

{table}

## Citation

```bibtex
{project.BIBTEX}
```
"""


def _load_scene(size_px: int) -> str:
    return f"""scene = preprocess({size_px})(Image.open("scene.jpg").convert("RGB")).unsqueeze(0)  # [1, 3, {size_px}, {size_px}]"""


def pretrained_card(*, repo: str, facts: PretrainingFacts, ablation: Ablation | None) -> str:
    dataset = DATASET_NAMES[facts.dataset]
    grid = facts.canvas_grid_size
    if ablation is None:
        title = f"CanViT-B, pretrained on {dataset}"
        summary = (f"CanViT-B pretrained on {dataset} by dense latent distillation from "
                   f"[{facts.teacher_repo}](https://huggingface.co/{facts.teacher_repo}), as in the paper.")
    else:
        row = f" ({ablation.paper_row})" if ablation.paper_row else ""
        title = f"CanViT pretraining ablation{row}: {ablation.paper_label}"
        summary = (f"One of the paper's pretraining ablations: a short pretraining run that changes one "
                   f"design choice of CanViT-B. Reproduce it with `python -m canvit_pytorch.pretrain --ablation "
                   f"{ablation.slug}`.")
    usage = f"""
import torch
from PIL import Image
from canvit_pytorch import CanViTForPretraining
from canvit_pytorch.episode import run_episode
from canvit_pytorch.policies import make_policy
from canvit_pytorch.preprocess import preprocess

model = CanViTForPretraining.from_pretrained("{repo}").eval()
{_load_scene(facts.scene_size_px)}

# Five glimpses of {facts.glimpse_size_px} px, coarse to fine: the full scene, then quadrants.
policy = make_policy("coarse_to_fine", batch_size=1, device=scene.device, num_glimpses=5, canvas_grid_size={grid})
with torch.inference_mode():
    steps = run_episode(
        canvit=model.canvit, images=scene, policy=policy, num_glimpses=5, glimpse_size_px={facts.glimpse_size_px},
        initial_state=model.init_state(batch_size=1, canvas_grid_size={grid}),
    )
    canvas = steps[-1].state.canvas
    features = model.canvit.canvas_patch_grid(canvas)  # [1, {grid}, {grid}, canvas_dim]: the scene-wide canvas
    teacher_patches = model.predict_teacher_patches(canvas)  # the teacher's features of the whole scene, standardized
"""
    details = [
        ("Pretraining data", dataset),
        ("Teacher", f"[{facts.teacher_repo}](https://huggingface.co/{facts.teacher_repo})"),
        ("Scenes", f"{facts.scene_size_px} px"),
        ("Glimpses", f"{facts.glimpse_size_px} px"),
        ("Canvas during pretraining", f"{grid} × {grid} (any size at inference)"),
    ]
    if facts.steps is not None:
        details.append(("Training steps", f"{facts.steps:,}"))
    data = ModelCardData(
        license="mit", library_name="canvit-pytorch", pipeline_tag="image-feature-extraction",
        tags=TAGS + (["ablation"] if ablation else []), datasets=[DATASET_HUB_IDS[facts.dataset]],
    )
    return _card(data=data, title=title, summary=summary, usage=usage, details=details)


def classifier_card(
    *, repo: str, pretrained_repo: str, details: list[tuple[str, str]], top1_accuracy: float, conditions: str,
) -> str:
    """conditions: how the reported accuracy was measured, e.g. "C2F, T=21, single run"."""
    usage = f"""
import torch
from PIL import Image
from canvit_pytorch import CanViTForImageClassification, Viewpoint, sample_at_viewpoint
from canvit_pytorch.benchmarks.imagenet import CLASS_NAMES
from canvit_pytorch.preprocess import preprocess

model = CanViTForImageClassification.from_pretrained("{repo}").eval()
{_load_scene(512)}

state = model.init_state(batch_size=1, canvas_grid_size=32)
with torch.inference_mode():
    viewpoint = Viewpoint.full_scene(batch_size=1, device=scene.device)
    glimpse = sample_at_viewpoint(spatial=scene, viewpoint=viewpoint, glimpse_size_px=128)
    logits, state = model(glimpse=glimpse, state=state, viewpoint=viewpoint)  # [1, 1000]; more glimpses refine it
print(CLASS_NAMES[logits.argmax().item()])
"""
    data = ModelCardData(
        license="mit", library_name="canvit-pytorch", pipeline_tag="image-classification",
        tags=TAGS, datasets=["imagenet-1k"], base_model=pretrained_repo, model_name=repo.split("/")[-1],
        eval_results=[EvalResult(
            task_type="image-classification", dataset_type="imagenet-1k", dataset_name="ImageNet-1k",
            dataset_split="validation", metric_type="accuracy", metric_name=f"Top-1 accuracy ({conditions})",
            metric_value=top1_accuracy,
        )],
    )
    summary = (f"[CanViT-B](https://huggingface.co/{pretrained_repo}) fine-tuned end to end for ImageNet-1k "
               f"classification by linear probing then fine-tuning (LP-FT), as in the paper: {top1_accuracy}% "
               f"top-1 accuracy on the validation set ({conditions}).")
    return _card(data=data, title="CanViT-B, fine-tuned for ImageNet-1k classification", summary=summary,
                 usage=usage, details=details)


def canvas_probe_card(*, repo: str, facts: CanvasProbeFacts) -> str:
    grid = facts.canvas_grid_size
    usage = f"""
import torch
from PIL import Image
from canvit_pytorch import CanViTForSemanticSegmentation, Viewpoint, sample_at_viewpoint
from canvit_pytorch.benchmarks.ade20k import CLASS_NAMES
from canvit_pytorch.preprocess import preprocess

model = CanViTForSemanticSegmentation.from_pretrained_with_probe(
    pretrained_repo="{facts.pretrained_repo}",
    probe_repo="{repo}",
).eval()
{_load_scene(facts.scene_size_px)}

state = model.init_state(batch_size=1, canvas_grid_size={grid})
with torch.inference_mode():
    viewpoint = Viewpoint.full_scene(batch_size=1, device=scene.device)
    glimpse = sample_at_viewpoint(spatial=scene, viewpoint=viewpoint, glimpse_size_px={facts.glimpse_size_px})
    logits, state = model(glimpse=glimpse, state=state, viewpoint=viewpoint)  # [1, 150, {grid}, {grid}]
labels = logits.argmax(dim=1)  # ADE20K classes, named in CLASS_NAMES
"""
    details = [
        ("Features", f"the {grid} × {grid} canvas of [{facts.pretrained_repo}](https://huggingface.co/{facts.pretrained_repo})"),
        ("Probe", "LayerNorm, dropout, BatchNorm, 1 × 1 convolution"),
        ("Training rollouts", (f"{facts.training_glimpses} glimpses of {facts.glimpse_size_px} px, "
                               f"{facts.training_policy} viewpoints, {facts.scene_size_px} px scenes")),
        *facts.training.rows(),
    ]
    data = ModelCardData(
        license="mit", library_name="canvit-pytorch", pipeline_tag="image-segmentation",
        tags=TAGS + ["ade20k", "linear-probe"], datasets=[ADE20K_HUB_ID], base_model=facts.pretrained_repo,
    )
    summary = (f"A linear ADE20K semantic segmentation probe on CanViT's {grid} × {grid} canvas, trained on "
               f"frozen features with the paper's probing protocol.")
    return _card(data=data, title=f"ADE20K probe on CanViT's {grid} × {grid} canvas", summary=summary,
                 usage=usage, details=details)


def dinov3_probe_card(*, repo: str, facts: DINOv3ProbeFacts) -> str:
    grid = facts.input_size_px // facts.patch_size
    usage = f"""
import torch
from PIL import Image
from canvit_pytorch.preprocess import preprocess
from canvit_pytorch.probes import SegmentationProbe
from canvit_pytorch.teacher import load_teacher

dinov3 = load_teacher("{facts.dinov3_repo}", torch.device("cpu"))
probe = SegmentationProbe.from_pretrained("{repo}").eval()
{_load_scene(facts.input_size_px)}

with torch.inference_mode():
    patches = dinov3(scene).patches.unflatten(1, ({grid}, {grid}))  # [1, {grid}, {grid}, {facts.embed_dim}]
    logits = probe(patches)  # [1, 150, {grid}, {grid}]
"""
    details = [
        ("Features", (f"[{facts.dinov3_repo}](https://huggingface.co/{facts.dinov3_repo}) patch features of "
                      f"{facts.input_size_px} px images")),
        ("Probe", "dropout, BatchNorm, 1 × 1 convolution"),
        *facts.training.rows(),
    ]
    data = ModelCardData(
        license="mit", library_name="canvit-pytorch", pipeline_tag="image-segmentation",
        tags=["dinov3", "ade20k", "linear-probe"], datasets=[ADE20K_HUB_ID], base_model=facts.dinov3_repo,
    )
    summary = (f"A linear ADE20K semantic segmentation probe on frozen {facts.dinov3_name} features at "
               f"{facts.input_size_px} px, a passive-vision reference for CanViT, trained with the paper's probing "
               f"protocol.")
    return _card(data=data, title=f"ADE20K probe on {facts.dinov3_name} at {facts.input_size_px} px",
                 summary=summary, usage=usage, details=details)
