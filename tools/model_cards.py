"""Generate concise native CanViT model cards and sanitized staging records."""

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from canvit_core.checkpoint import FORMAT
from canvit_pytorch import project

Backend = Literal["mlx", "nnx"]
ARCHIVE_TAG = f"before-format-{FORMAT[:8]}"


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


def _backend_fragments(backend: Backend) -> tuple[str, str, str, str]:
    if backend == "nnx":
        return (
            "import jax.numpy as jnp\nfrom canvit_nnx import Viewpoint, sample_at_viewpoint",
            "from canvit_nnx.preprocess import preprocess",
            "jnp.asarray",
            "canvit_nnx",
        )
    return (
        "import mlx.core as mx\nfrom canvit_mlx import Viewpoint, sample_at_viewpoint",
        "from canvit_mlx.preprocess import preprocess",
        "mx.array",
        "canvit_mlx",
    )


def _model_usage(*, backend: Backend, repo: str, model_name: str, scene_size_px: int, glimpse_size_px: int, canvas_grid_size: int) -> str:
    imports, preprocess_import, array_constructor, module_name = _backend_fragments(backend)
    if model_name == "CanViTForPretraining":
        model_import = "CanViTForPretraining"
        result = "output = model(glimpse=glimpse, state=state, viewpoint=viewpoint)\nstate = output.state"
    else:
        model_import = "CanViTForImageClassification"
        result = "logits, state = model(glimpse=glimpse, state=state, viewpoint=viewpoint)"
    return f'''{imports}
from PIL import Image
from {module_name} import {model_import}
{preprocess_import}

scene = {array_constructor}(preprocess({scene_size_px})(Image.open("scene.jpg").convert("RGB")))[None]
model = {model_import}.from_pretrained("{repo}")
model.eval()
state = model.init_state(batch_size=1, canvas_grid_size={canvas_grid_size})
viewpoint = Viewpoint.full_scene(batch_size=1)
glimpse = sample_at_viewpoint(spatial=scene, viewpoint=viewpoint, glimpse_size_px={glimpse_size_px})
{result}'''


def _probe_usage(entry: ProbeEntry, *, num_classes: int) -> str:
    imports, preprocess_import, array_constructor, module_name = _backend_fragments(entry.backend)
    evaluation = "\nmx.eval(logits)" if entry.backend == "mlx" else ""
    return f'''{imports}
from PIL import Image
from {module_name} import CanViTForSemanticSegmentation
{preprocess_import}

scene = {array_constructor}(preprocess({entry.scene_size_px})(Image.open("scene.jpg").convert("RGB")))[None]
model = CanViTForSemanticSegmentation.from_pretrained_with_probe(
    pretrained_repo="{entry.pretrained_native_repo}",
    probe_repo="{entry.target_repo}",
)
model.eval()
state = model.init_state(batch_size=1, canvas_grid_size={entry.canvas_grid_size})
viewpoint = Viewpoint.full_scene(batch_size=1)
glimpse = sample_at_viewpoint(spatial=scene, viewpoint=viewpoint, glimpse_size_px={entry.glimpse_size_px})
logits, state = model(glimpse=glimpse, state=state, viewpoint=viewpoint)  # [1, {entry.canvas_grid_size}, {entry.canvas_grid_size}, {num_classes}]{evaluation}'''


def _details(rows: list[tuple[str, str]]) -> str:
    return "| | |\n|---|---|\n" + "\n".join(f"| {name} | {value} |" for name, value in rows)


def _model_card(entry: ModelEntry) -> tuple[str, dict[str, Any]]:
    record = _read_json(entry.artifact / "config.json")
    conversion = _read_json(entry.artifact / "conversion.json")
    if record.get("format") != FORMAT:
        raise ValueError(f"{entry.artifact}: unexpected format {record.get('format')!r}")
    assert record["backend"] == entry.backend, (record["backend"], entry.backend)
    model_name = record["model"]
    assert model_name in {"CanViTForPretraining", "CanViTForImageClassification"}, model_name
    label = _backend_label(entry.backend)
    is_pretraining = model_name == "CanViTForPretraining"
    scene_size_px = int(conversion["scene_size_px"])
    glimpse_size_px = int(conversion["glimpse_size_px"])
    canvas_grid_size = int(conversion["canvas_grid_size"])
    title = f"CanViT {'pretraining checkpoint' if is_pretraining else 'classifier'} ({label})"
    summary = f"A {label} CanViT {'pretraining' if is_pretraining else 'image-classification'} checkpoint."
    usage = _model_usage(
        backend=entry.backend,
        repo=entry.target_repo,
        model_name=model_name,
        scene_size_px=scene_size_px,
        glimpse_size_px=glimpse_size_px,
        canvas_grid_size=canvas_grid_size,
    )
    rows = [
        ("Backend", label),
        ("Model class", f"`{model_name}`"),
        ("Source checkpoint", _hub_link(entry.source_repo, entry.source_revision)),
        ("Validation geometry", f"{scene_size_px} px scenes, {glimpse_size_px} px glimpses, {canvas_grid_size} × {canvas_grid_size} canvas"),
    ]
    task_config = record["config"]
    if "n_classes" in task_config:
        rows.append(("Classes", str(task_config["n_classes"])))
    frontmatter: dict[str, str | list[str]] = {
        "library_name": _package_name(entry.backend),
        "pipeline_tag": "image-feature-extraction" if is_pretraining else "image-classification",
        "tags": ["canvit", "active-vision", "vision-transformer", entry.backend],
        "license": "mit",
    }
    body = "\n\n".join((
        _frontmatter(frontmatter),
        f"# {title}",
        summary,
        project.DESCRIPTION,
        f"[Paper ({project.VENUE})]({project.PAPER_URL}) · [Code]({project.CODE_URL}) · [Project page]({project.PAGE_URL}) · [All checkpoints]({project.HUB_ORG_URL})",
        "## Install\n\n```bash\n" + _install(entry.backend) + "\n```",
        "## Usage\n\n```python\n" + usage + "\n```",
        "## Details\n\n" + _details(rows),
        "## Citation\n\n```bibtex\n" + project.BIBTEX + "\n```",
    )) + "\n"
    return body, _provenance(entry, record, conversion)


def _probe_card(entry: ProbeEntry) -> tuple[str, dict[str, Any]]:
    record = _read_json(entry.artifact / "config.json")
    conversion = _read_json(entry.artifact / "conversion.json")
    if record.get("format") != FORMAT:
        raise ValueError(f"{entry.artifact}: unexpected format {record.get('format')!r}")
    label = _backend_label(entry.backend)
    assert record["backend"] == entry.backend and record["model"] == "SegmentationProbe", record
    frontmatter: dict[str, str | list[str]] = {
        "library_name": _package_name(entry.backend),
        "pipeline_tag": "image-segmentation",
        "tags": ["canvit", "active-vision", "vision-transformer", "ade20k", "linear-probe", entry.backend],
        "license": "mit",
    }
    rows = [
        ("Backend", label),
        ("Model class", "`SegmentationProbe`"),
        ("Source probe", _hub_link(entry.source_repo, entry.source_revision)),
        ("CanViT checkpoint", _hub_link(entry.pretrained_native_repo)),
        ("Classes", str(record["config"]["num_classes"])),
        ("Example geometry", f"{entry.scene_size_px} px scenes, {entry.glimpse_size_px} px glimpses, {entry.canvas_grid_size} × {entry.canvas_grid_size} canvas"),
    ]
    body = "\n\n".join((
        _frontmatter(frontmatter),
        f"# ADE20K probe on CanViT's canvas ({label})",
        f"A {label} ADE20K segmentation probe for CanViT canvas features. Load it alongside the separately published CanViT checkpoint.",
        project.DESCRIPTION,
        f"[Paper ({project.VENUE})]({project.PAPER_URL}) · [Code]({project.CODE_URL}) · [Project page]({project.PAGE_URL}) · [All checkpoints]({project.HUB_ORG_URL})",
        "## Install\n\n```bash\n" + _install(entry.backend) + "\n```",
        "## Usage\n\n```python\n" + _probe_usage(entry, num_classes=record["config"]["num_classes"]) + "\n```",
        "## Details\n\n" + _details(rows),
        "## Citation\n\n```bibtex\n" + project.BIBTEX + "\n```",
    )) + "\n"
    return body, _provenance(entry, record, conversion)


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
    result: dict[str, Any] = {
        "source": source,
        "target": {
            "backend": record["backend"],
            "model": record["model"],
            "format": record["format"],
            "config_sha256": _sha256(entry.artifact / "config.json"),
            "weights_sha256": _sha256(entry.artifact / "model.safetensors"),
        },
        "validation": validation,
    }
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
    card, provenance = _model_card(entry)
    return _stage(entry.artifact, destination_root / entry.target_repo.rsplit("/", 1)[-1], card, provenance)


def stage_probe(entry: ProbeEntry, destination_root: Path) -> Path:
    card, provenance = _probe_card(entry)
    return _stage(entry.artifact, destination_root / entry.target_repo.rsplit("/", 1)[-1], card, provenance)


__all__ = ["ModelEntry", "ProbeEntry", "stage_model", "stage_probe"]
