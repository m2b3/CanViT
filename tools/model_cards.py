"""Model cards and staging records of native CanViT checkpoints (canvit-nnx, canvit-mlx): converted pretraining
checkpoints and classifiers, probes, and canvit-nnx's COCO detectors.

A card is rendered from what its checkpoint's records state: the staging functions read a conversion's output
directory, tools.refresh_cards reads a published repo's config.json and provenance.json.
"""

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from canvit_core.checkpoint import FORMAT
from canvit_pytorch import project
from canvit_pytorch.hub.cards import details_table

Backend = Literal["mlx", "nnx"]
ARCHIVE_TAG = f"before-format-{FORMAT[:8]}"
TAGS = ["canvit", "active-vision", "vision-transformer"]


@dataclass(frozen=True)
class Geometry:
    scene_size_px: int
    glimpse_size_px: int
    canvas_grid_size: int


@dataclass(frozen=True)
class Source:
    """The published checkpoint a native one was converted from, at the revision converted."""

    repo: str
    revision: str


@dataclass(frozen=True)
class NativeTraining:
    """What the card of a checkpoint trained outside canvit-pytorch states, since no PyTorch repo states it."""

    title: str
    summary: str
    rows: list[tuple[str, str]]
    results: str | None = None
    base_model: str | None = None
    """The checkpoint it was trained from, when there is one."""
    datasets: tuple[str, ...] = ()


@dataclass(frozen=True)
class NativeModel:
    """A pretraining or classification checkpoint in a native format, as its card states it."""

    repo: str
    backend: Backend
    record: dict[str, Any]
    """Its config.json."""
    geometry: Geometry
    """Of the usage example; for a conversion, where it was checked against its source."""
    source: Source | None
    training: NativeTraining | None = None


@dataclass(frozen=True)
class NativeProbe:
    repo: str
    backend: Backend
    record: dict[str, Any]
    pretrained_native_repo: str
    geometry: Geometry
    """The geometry of the usage example: the probe's training geometry."""
    source: Source | None
    training: NativeTraining | None = None


@dataclass(frozen=True)
class ModelEntry:
    artifact: Path
    target_repo: str
    backend: Backend
    source_repo: str
    source_revision: str
    old_repo: str | None = None
    old_revision: str | None = None


@dataclass(frozen=True)
class ProbeEntry:
    artifact: Path
    target_repo: str
    backend: Backend
    source_repo: str
    source_revision: str
    pretrained_native_repo: str
    scene_size_px: int
    glimpse_size_px: int
    canvas_grid_size: int


@dataclass(frozen=True)
class DetectorEntry:
    """A CanViTForObjectDetection in canvit-nnx's format, with the records its card and provenance are made of."""

    artifact: Path  # config.json and model.safetensors
    target_repo: str
    # pretrained, checkpoint_step, total_steps, scene_size_px, glimpse_size_px, recipe (detect.train.Recipe)
    training: dict[str, Any]
    evaluation: dict[str, Any]  # split, images, policy, repeats, glimpses, statistics: {"<stat>_t<glimpse>": {mean, ci95}}
    validation: dict[str, Any]  # the converted detector against the implementation that trained it


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _frontmatter(fields: dict[str, str | list[str]]) -> str:
    lines = ["---"]
    for name, value in fields.items():
        if isinstance(value, list):
            lines.append(f"{name}:")
            lines.extend(f"  - {_yaml_string(item)}" for item in value)
        else:
            lines.append(f"{name}: {_yaml_string(value)}")
    lines.append("---")
    return "\n".join(lines)


def _hub_link(repo: str, revision: str | None = None) -> str:
    if revision is None:
        return f"[{repo}](https://huggingface.co/{repo})"
    return f"[{repo}](https://huggingface.co/{repo}/tree/{revision}) at `{revision}`"


def _backend_label(backend: Backend) -> str:
    return {"nnx": "JAX / Flax NNX", "mlx": "MLX"}[backend]


def _package_name(backend: Backend) -> str:
    return f"canvit-{backend}"


def _install(backend: Backend) -> str:
    return f"uv add {_package_name(backend)}"


def _checked_record(record: dict[str, Any], backend: Backend, models: set[str], where: object) -> str:
    if record.get("format") != FORMAT:
        raise ValueError(f"{where}: unexpected format {record.get('format')!r}")
    assert record["backend"] == backend and record["model"] in models, (where, record["backend"], record["model"])
    return record["model"]


def _card(*, frontmatter: dict[str, str | list[str]], title: str, summary: str, backend: Backend, usage: str,
          rows: list[tuple[str, str]], results: str | None = None) -> str:
    return "\n\n".join((
        _frontmatter(frontmatter),
        f"# {title}",
        summary,
        project.DESCRIPTION,
        f"[Paper ({project.VENUE})]({project.PAPER_URL}) · [Code]({project.CODE_URL}) · [Project page]({project.PAGE_URL}) · [All checkpoints]({project.HUB_ORG_URL})",
        "## Install\n\n```bash\n" + _install(backend) + "\n```",
        "## Usage\n\n```python\n" + usage + "\n```",
        *([f"## Results\n\n{results}"] if results is not None else []),
        "## Details\n\n" + details_table(rows),
        f"## License\n\n{project.CHECKPOINT_LICENSE}",
        "## Citation\n\n```bibtex\n" + project.BIBTEX + "\n```",
    )) + "\n"


def _backend_fragments(backend: Backend) -> tuple[str, str, str, str]:
    if backend == "nnx":
        return (
            "import jax.numpy as jnp",
            "from canvit_nnx.preprocess import preprocess",
            "jnp.asarray",
            "canvit_nnx",
        )
    return (
        "import mlx.core as mx",
        "from canvit_mlx.preprocess import preprocess",
        "mx.array",
        "canvit_mlx",
    )


def _model_usage(*, backend: Backend, repo: str, model_name: str, geometry: Geometry) -> str:
    imports, preprocess_import, array_constructor, module_name = _backend_fragments(backend)
    if model_name == "CanViTForPretraining":
        result = "output = model(glimpse=glimpse, state=state, viewpoint=viewpoint)\nstate = output.state"
    else:
        result = "logits, state = model(glimpse=glimpse, state=state, viewpoint=viewpoint)"
    return f'''{imports}
from PIL import Image

from {module_name} import {model_name}, Viewpoint, sample_at_viewpoint
{preprocess_import}

scene = {array_constructor}(preprocess({geometry.scene_size_px})(Image.open("scene.jpg").convert("RGB")))[None]
model = {model_name}.from_pretrained("{repo}")
model.eval()
state = model.init_state(batch_size=1, canvas_grid_size={geometry.canvas_grid_size})
viewpoint = Viewpoint.full_scene(batch_size=1)
glimpse = sample_at_viewpoint(spatial=scene, viewpoint=viewpoint, glimpse_size_px={geometry.glimpse_size_px})
{result}'''


def _probe_usage(probe: NativeProbe, *, num_classes: int) -> str:
    imports, preprocess_import, array_constructor, module_name = _backend_fragments(probe.backend)
    evaluation = "\nmx.eval(logits)" if probe.backend == "mlx" else ""
    g = probe.geometry
    return f'''{imports}
from PIL import Image

from {module_name} import CanViTForSemanticSegmentation, Viewpoint, sample_at_viewpoint
{preprocess_import}

scene = {array_constructor}(preprocess({g.scene_size_px})(Image.open("scene.jpg").convert("RGB")))[None]
model = CanViTForSemanticSegmentation.from_pretrained_with_probe(
    pretrained_repo="{probe.pretrained_native_repo}",
    probe_repo="{probe.repo}",
)
model.eval()
state = model.init_state(batch_size=1, canvas_grid_size={g.canvas_grid_size})
viewpoint = Viewpoint.full_scene(batch_size=1)
glimpse = sample_at_viewpoint(spatial=scene, viewpoint=viewpoint, glimpse_size_px={g.glimpse_size_px})
logits, state = model(glimpse=glimpse, state=state, viewpoint=viewpoint)  # [1, {g.canvas_grid_size}, {g.canvas_grid_size}, {num_classes}]{evaluation}'''


def _detector_usage(*, repo: str, scene_size_px: int, glimpse_size_px: int, canvas_grid_size: int, num_glimpses: int) -> str:
    return f'''import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from canvit_nnx import sample_at_viewpoint
from canvit_nnx.detect.data import normalize
from canvit_nnx.detect.decode import decode
from canvit_nnx.detect.model import CanViTForObjectDetection
from canvit_nnx.policies import coarse_to_fine

image = Image.open("scene.jpg").convert("RGB").resize(({scene_size_px}, {scene_size_px}), Image.Resampling.BILINEAR)
scene = normalize(jnp.asarray(np.asarray(image))[None])
model = nnx.view(CanViTForObjectDetection.from_pretrained("{repo}"), use_running_average=True)
state = model.init_state(batch_size=1, canvas_grid_size={canvas_grid_size})
for viewpoint in coarse_to_fine(jax.random.key(0), batch_size=1, num_glimpses={num_glimpses}):
    glimpse = sample_at_viewpoint(spatial=scene, viewpoint=viewpoint, glimpse_size_px={glimpse_size_px})
    maps, state = model(glimpse=glimpse, state=state, viewpoint=viewpoint)
    detections = decode(maps)  # normalized (ymin, xmin, ymax, xmax) boxes, scores and COCO class indices'''


def _geometry_text(g: Geometry) -> str:
    return f"{g.scene_size_px} px scenes, {g.glimpse_size_px} px glimpses, {g.canvas_grid_size} × {g.canvas_grid_size} canvas"


def _conversion_rows(source: Source | None, checked_at: Geometry | None) -> list[tuple[str, str]]:
    if source is None:
        return []
    rows = [("Converted from", _hub_link(source.repo, source.revision))]
    return rows + ([("Conversion checked at", _geometry_text(checked_at))] if checked_at is not None else [])


def _training_frontmatter(training: NativeTraining | None, source: Source | None) -> dict[str, str | list[str]]:
    base = training.base_model if training is not None and training.base_model else (source.repo if source else None)
    fields: dict[str, str | list[str]] = {"base_model": base} if base else {}
    if training is not None and training.datasets:
        fields["datasets"] = list(training.datasets)
    return fields


def model_card(model: NativeModel) -> str:
    model_name = _checked_record(model.record, model.backend, {"CanViTForPretraining", "CanViTForImageClassification"},
                                 model.repo)
    label, package = _backend_label(model.backend), _package_name(model.backend)
    is_pretraining = model_name == "CanViTForPretraining"
    rows = [("Backend", label), ("Model class", f"`{model_name}`"), *_conversion_rows(model.source, model.geometry)]
    if "n_classes" in model.record["config"]:
        rows.append(("Classes", str(model.record["config"]["n_classes"])))
    rows += model.training.rows if model.training is not None else []
    frontmatter: dict[str, str | list[str]] = {
        "library_name": package,
        "pipeline_tag": "image-feature-extraction" if is_pretraining else "image-classification",
        "tags": [*TAGS, model.backend],
        "license": "mit",
        **_training_frontmatter(model.training, model.source),
    }
    if (training := model.training) is None:
        assert model.source is not None, model.repo
        title = f"CanViT {'pretraining checkpoint' if is_pretraining else 'classifier'} ({label})"
        summary = f"{_hub_link(model.source.repo)} in the checkpoint format of {package}, for {label}."
    else:
        title, summary = f"{training.title} ({label})", training.summary
    usage = _model_usage(backend=model.backend, repo=model.repo, model_name=model_name, geometry=model.geometry)
    return _card(frontmatter=frontmatter, title=title, summary=summary, backend=model.backend, usage=usage, rows=rows,
                 results=training.results if training else None)


def probe_card(probe: NativeProbe) -> str:
    _checked_record(probe.record, probe.backend, {"SegmentationProbe"}, probe.repo)
    label, package = _backend_label(probe.backend), _package_name(probe.backend)
    rows = [("Backend", label), ("Model class", "`SegmentationProbe`"), *_conversion_rows(probe.source, None),
            ("CanViT checkpoint", _hub_link(probe.pretrained_native_repo)),
            ("Classes", str(probe.record["config"]["num_classes"]))]
    rows += probe.training.rows if probe.training is not None else [("Example geometry", _geometry_text(probe.geometry))]
    frontmatter: dict[str, str | list[str]] = {
        "library_name": package,
        "pipeline_tag": "image-segmentation",
        "tags": [*TAGS, "ade20k", "linear-probe", probe.backend],
        "license": "mit",
        **_training_frontmatter(probe.training, probe.source),
    }
    if (training := probe.training) is None:
        assert probe.source is not None, probe.repo
        title = f"ADE20K probe on CanViT's canvas ({label})"
        summary = (f"{_hub_link(probe.source.repo)} in the checkpoint format of {package}: an ADE20K segmentation "
                   "probe for CanViT's canvas, loaded alongside the separately published CanViT checkpoint.")
    else:
        title, summary = f"{training.title} ({label})", training.summary
    return _card(frontmatter=frontmatter, title=title, summary=summary, backend=probe.backend,
                 usage=_probe_usage(probe, num_classes=probe.record["config"]["num_classes"]), rows=rows,
                 results=training.results if training else None)


def _detector_card(entry: "DetectorEntry") -> tuple[str, dict[str, Any]]:
    record = _read_json(entry.artifact / "config.json")
    _checked_record(record, "nnx", {"CanViTForObjectDetection"}, entry.artifact)
    training, evaluation = entry.training, entry.evaluation
    recipe, statistics = training["recipe"], evaluation["statistics"]
    glimpses = range(evaluation["glimpses"])

    def cell(name: str, glimpse: int) -> str:
        value = statistics[f"{name}_t{glimpse}"]
        interval = value.get("ci95")
        if interval is None or interval[0] >= interval[1]:
            return f"{value['mean']:.4f}"
        return f"{value['mean']:.4f} [{interval[0]:.4f}, {interval[1]:.4f}]"

    results = (
        f"COCO {evaluation['split']} ({evaluation['images']} images), {evaluation['policy']} viewpoints: means over "
        f"{evaluation['repeats']} independent viewpoint draws, with 95% bootstrap intervals over the draws where they "
        "vary.\n\n| Glimpse | AP | AP50 | AP75 |\n|---|---|---|---|\n" + "\n".join(
            f"| {glimpse} | {cell('AP', glimpse)} | {cell('AP50', glimpse)} | {cell('AP75', glimpse)} |"
            for glimpse in glimpses
        )
    )
    rows = [
        ("Backend", _backend_label("nnx")),
        ("Model class", "`CanViTForObjectDetection`"),
        ("Starting checkpoint", training["pretrained"]),
        ("Training", (
            f"step {training['checkpoint_step']} of {training['total_steps']}, {recipe['batch_size']} COCO "
            f"train2017 scenes per step, {recipe['num_glimpses']} glimpses per rollout; the recipe is in "
            "provenance.json"
        )),
        ("Geometry", (
            f"{training['scene_size_px']} px scenes, {training['glimpse_size_px']} px glimpses, "
            f"{recipe['canvas_grid_size']} × {recipe['canvas_grid_size']} canvas"
        )),
        ("Classes", str(record["config"]["num_classes"])),
    ]
    frontmatter: dict[str, str | list[str]] = {
        "library_name": _package_name("nnx"),
        "pipeline_tag": "object-detection",
        "tags": [*TAGS, "coco", "centernet", "nnx"],
        "license": "mit",
    }
    usage = _detector_usage(
        repo=entry.target_repo,
        scene_size_px=training["scene_size_px"],
        glimpse_size_px=training["glimpse_size_px"],
        canvas_grid_size=recipe["canvas_grid_size"],
        num_glimpses=evaluation["glimpses"],
    )
    summary = ("CanViT-B with a CenterNet head on its canvas, trained end to end for COCO 2017 object detection; it "
               "predicts boxes after every glimpse.")
    body = _card(frontmatter=frontmatter, title="CanViT COCO detector (JAX / Flax NNX)", summary=summary,
                 backend="nnx", usage=usage, rows=rows, results=results)
    provenance = {
        "target": _target(entry.artifact, record),
        "training": training,
        "evaluation": evaluation,
        "validation": entry.validation,
    }
    return body, provenance


def _target(artifact: Path, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "backend": record["backend"],
        "model": record["model"],
        "format": record["format"],
        "config_sha256": _sha256(artifact / "config.json"),
        "weights_sha256": _sha256(artifact / "model.safetensors"),
    }


def _provenance(entry: ModelEntry | ProbeEntry, record: dict[str, Any], conversion: dict[str, Any]) -> dict[str, Any]:
    source: dict[str, Any] = {"repo": entry.source_repo, "revision": entry.source_revision}
    source_sidecar = entry.artifact / "source.json"
    if source_sidecar.is_file():
        source["files"] = _read_json(source_sidecar).get("source_files", {})
    validation: dict[str, Any] = {
        "metrics_before_save": conversion["metrics_before_save"],
        "metrics_after_reload": conversion["metrics_after_reload"],
    }
    for key in ("canvas_grid_size", "glimpse_size_px", "scene_size_px", "steps", "verification_image_sha256"):
        if key in conversion:
            validation[key] = conversion[key]
    result: dict[str, Any] = {"source": source, "target": _target(entry.artifact, record), "validation": validation}
    if isinstance(entry, ModelEntry) and entry.old_repo is not None:
        result["old_format_archive"] = {"repo": entry.old_repo, "revision": entry.old_revision, "tag": ARCHIVE_TAG}
    if isinstance(entry, ProbeEntry):
        result["pretrained_native_repo"] = entry.pretrained_native_repo
    return result


def _stage(artifact: Path, destination: Path, card: str, provenance: dict[str, Any]) -> Path:
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    shutil.copyfile(artifact / "config.json", destination / "config.json")
    shutil.copyfile(artifact / "model.safetensors", destination / "model.safetensors")
    (destination / "README.md").write_text(card)
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return destination


def stage_model(entry: ModelEntry, destination_root: Path) -> Path:
    record = _read_json(entry.artifact / "config.json")
    conversion = _read_json(entry.artifact / "conversion.json")
    geometry = Geometry(scene_size_px=int(conversion["scene_size_px"]), glimpse_size_px=int(conversion["glimpse_size_px"]),
                        canvas_grid_size=int(conversion["canvas_grid_size"]))
    card = model_card(NativeModel(repo=entry.target_repo, backend=entry.backend, record=record, geometry=geometry,
                                  source=Source(entry.source_repo, entry.source_revision)))
    return _stage(entry.artifact, destination_root / entry.target_repo.rsplit("/", 1)[-1], card,
                  _provenance(entry, record, conversion))


def stage_probe(entry: ProbeEntry, destination_root: Path) -> Path:
    record = _read_json(entry.artifact / "config.json")
    conversion = _read_json(entry.artifact / "conversion.json")
    card = probe_card(NativeProbe(
        repo=entry.target_repo, backend=entry.backend, record=record, pretrained_native_repo=entry.pretrained_native_repo,
        geometry=Geometry(entry.scene_size_px, entry.glimpse_size_px, entry.canvas_grid_size),
        source=Source(entry.source_repo, entry.source_revision),
    ))
    return _stage(entry.artifact, destination_root / entry.target_repo.rsplit("/", 1)[-1], card,
                  _provenance(entry, record, conversion))


def stage_detector(entry: "DetectorEntry", destination_root: Path) -> Path:
    card, provenance = _detector_card(entry)
    return _stage(entry.artifact, destination_root / entry.target_repo.rsplit("/", 1)[-1], card, provenance)


__all__ = [
    "DetectorEntry", "Geometry", "ModelEntry", "NativeModel", "NativeProbe", "NativeTraining", "ProbeEntry", "Source",
    "model_card", "probe_card", "stage_detector", "stage_model", "stage_probe",
]
