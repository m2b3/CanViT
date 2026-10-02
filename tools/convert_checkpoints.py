import hashlib
import importlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import tyro
from canvit_core import CanViTConfig
from canvit_core.preprocess import preprocess
from canvit_pytorch import (
    CanViTForImageClassification as TorchClassifier,
)
from canvit_pytorch import (
    CanViTForPretraining as TorchPretraining,
)
from canvit_pytorch import (
    Viewpoint as TorchViewpoint,
)
from canvit_pytorch import (
    sample_at_viewpoint as torch_sample_at_viewpoint,
)
from canvit_pytorch.hub.repos import RELEASED_CANVAS_GRID_SIZE, RELEASED_GLIMPSE_SIZE_PX, RELEASED_SCENE_SIZE_PX
from canvit_pytorch.preprocess import preprocess as torch_preprocess
from canvit_pytorch.probes import SegmentationProbe as TorchSegmentationProbe
from huggingface_hub import snapshot_download
from PIL import Image

from tools.checkpoint_conversion import (
    BackendName,
    convert_named_arrays,
    load_native_arrays,
    native_state_arrays,
)

log = logging.getLogger(__name__)
ModelKind = Literal["classification", "pretraining", "probe"]


@dataclass(frozen=True)
class ConvertCheckpoints:
    source: str
    output: Path
    backend: BackendName
    revision: str | None = None
    image: Path | None = None
    scene_size_px: int = RELEASED_SCENE_SIZE_PX
    kind: ModelKind | None = None
    canvas_grid_size: int = RELEASED_CANVAS_GRID_SIZE
    glimpse_size_px: int = RELEASED_GLIMPSE_SIZE_PX
    steps: int = 3


def _source_snapshot(source: str, revision: str | None) -> tuple[Path, str]:
    path = Path(source)
    if path.is_dir():
        digest = hashlib.sha256()
        for filename in ("config.json", "model.safetensors"):
            file = path / filename
            if not file.is_file():
                raise FileNotFoundError(f"local checkpoint is missing {file}")
            digest.update(filename.encode())
            with file.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
        return path, f"local:{digest.hexdigest()}"
    snapshot = Path(snapshot_download(source, revision=revision, allow_patterns=["config.json", "model.safetensors"]))
    return snapshot, snapshot.name


def _read_config(source: Path) -> dict[str, Any]:
    record = json.loads((source / "config.json").read_text())
    if not isinstance(record, dict):
        raise ValueError(f"{source}: config.json must contain an object")
    return record


def _model_kind(record: dict[str, Any], requested: ModelKind | None) -> ModelKind:
    inferred: ModelKind
    probe_fields = {"embed_dim", "num_classes", "dropout", "use_ln"}
    if probe_fields <= record.keys():
        inferred = "probe"
    elif "teacher_dim" in record and "teacher_patch_grid" in record:
        inferred = "pretraining"
    elif "n_classes" in record:
        inferred = "classification"
    else:
        raise ValueError("checkpoint config does not identify a CanViT pretraining, classification, or probe model")
    if requested is not None and requested != inferred:
        raise ValueError(f"requested {requested} conversion for a {inferred} checkpoint")
    return inferred


def _load_torch_model(source: Path, kind: ModelKind) -> torch.nn.Module:
    cls = {
        "classification": TorchClassifier,
        "pretraining": TorchPretraining,
        "probe": TorchSegmentationProbe,
    }[kind]
    model = cls.from_pretrained(str(source))
    return model.eval()


def _native_module(backend: BackendName) -> Any:
    return importlib.import_module(f"canvit_{backend}")


def _native_model(module: Any, backend: BackendName, kind: ModelKind, config: dict[str, Any]) -> Any:
    if kind == "probe":
        kwargs = {name: config[name] for name in ("embed_dim", "num_classes", "dropout", "use_ln")}
        cls = module.SegmentationProbe
    else:
        kwargs: dict[str, Any] = {"canvit_config": CanViTConfig(**config["canvit_config"])}
        if kind == "pretraining":
            kwargs |= {"teacher_dim": config["teacher_dim"], "teacher_patch_grid": config["teacher_patch_grid"]}
            cls = module.CanViTForPretraining
        else:
            kwargs["n_classes"] = config["n_classes"]
            cls = module.CanViTForImageClassification
    if backend == "nnx":
        from flax import nnx

        kwargs["rngs"] = nnx.Rngs(0)
    return cls(**kwargs)


def _load_converted_weights(torch_model: torch.nn.Module, native_model: Any, backend: BackendName) -> None:
    source_arrays = {name: value.detach().cpu().numpy() for name, value in torch_model.state_dict().items()}
    target_arrays = native_state_arrays(native_model, backend=backend)
    converted = convert_named_arrays(
        source_arrays,
        target_shapes={name: tuple(value.shape) for name, value in target_arrays.items()},
        backend=backend,
    )
    load_native_arrays(native_model, converted, backend=backend)


def _set_native_probe_eval(probe: Any, backend: BackendName) -> None:
    probe.eval()
    if backend == "nnx":
        probe.bn.use_running_average = True


def verify_probe_outputs(
    torch_model: torch.nn.Module,
    native_model: Any,
    *,
    backend: BackendName,
) -> dict[str, dict[str, float]]:
    if not isinstance(torch_model, TorchSegmentationProbe):
        raise TypeError(f"expected PyTorch SegmentationProbe, got {type(torch_model).__name__}")
    _set_native_probe_eval(native_model, backend)
    features = np.random.default_rng(9018).normal(
        size=(2, 5, 7, torch_model.embed_dim),
    ).astype(np.float32)
    with torch.no_grad():
        reference = torch_model(torch.from_numpy(features)).permute(0, 2, 3, 1).numpy()
    native = np.asarray(native_model(_native_array(features, backend)))
    if not np.isfinite(reference).all() or not np.isfinite(native).all():
        raise FloatingPointError("non-finite segmentation probe logits")
    return {"probe.logits": _metric(reference, native)}


def _run_probe_conversion(
    config: ConvertCheckpoints,
    *,
    source: Path,
    source_revision: str,
    record: dict[str, Any],
) -> None:
    if config.image is not None:
        raise ValueError("--image is only valid for CanViT model conversion")
    torch_model = _load_torch_model(source, "probe")
    native_model = _native_model(_native_module(config.backend), config.backend, "probe", record)
    _load_converted_weights(torch_model, native_model, config.backend)
    metrics_before_save = verify_probe_outputs(torch_model, native_model, backend=config.backend)
    _assert_metrics(metrics_before_save)
    for name, values in metrics_before_save.items():
        log.info("%s max_abs=%.3e relative_l2=%.3e normalized_rmse=%.3e", name, values["max_abs"],
                 values["relative_l2"], values["normalized_rmse"])
    native_model.save_pretrained(config.output)
    reloaded = type(native_model).from_pretrained(config.output)
    metrics_after_reload = verify_probe_outputs(torch_model, reloaded, backend=config.backend)
    _assert_metrics(metrics_after_reload)
    (config.output / "conversion.json").write_text(json.dumps({
        "source": config.source,
        "source_revision": source_revision,
        "backend": config.backend,
        "kind": "probe",
        "metrics_before_save": metrics_before_save,
        "metrics_after_reload": metrics_after_reload,
    }, indent=2, sort_keys=True) + "\n")
    log.info("wrote %s probe checkpoint to %s", config.backend, config.output)


def _native_array(value: np.ndarray, backend: BackendName) -> Any:
    if backend == "mlx":
        import mlx.core as mx

        return mx.array(value)
    import jax.numpy as jnp

    return jnp.asarray(value)


def _native_viewpoint(module: Any, backend: BackendName, centers: np.ndarray, scales: np.ndarray) -> Any:
    return module.Viewpoint(centers=_native_array(centers, backend), scales=_native_array(scales, backend))


def _as_numpy(value: Any, backend: BackendName) -> np.ndarray:
    if backend == "mlx":
        import mlx.core as mx

        mx.eval(value)
    return np.asarray(value)


def _teacher_values(model: Any, state: Any) -> dict[str, Any]:
    return {
        "teacher_patches": model.predict_teacher_patches(state.canvas),
        "teacher_cls": model.predict_teacher_cls(state.recurrent_cls),
    }


def _split_result(result: Any, kind: ModelKind) -> tuple[dict[str, Any], Any]:
    if kind == "classification":
        logits, state = result
        return {"logits": logits, "canvas": state.canvas, "recurrent_cls": state.recurrent_cls}, state
    state = result.state
    values = {
        "canvas": state.canvas,
        "recurrent_cls": state.recurrent_cls,
        "glimpse_patches": result.glimpse_patches,
    }
    if result.vpe is not None:
        values["vpe"] = result.vpe
    return values, state


def _native_step(
    model: Any,
    kind: ModelKind,
    *,
    glimpse: Any,
    state: Any,
    viewpoint: Any,
    backend: BackendName,
) -> tuple[dict[str, np.ndarray], Any]:
    result = model(glimpse=glimpse, state=state, viewpoint=viewpoint)
    values, state = _split_result(result, kind)
    if kind == "pretraining":
        values |= _teacher_values(model, state)
    if backend == "mlx":
        import mlx.core as mx

        mx.eval(list(values.values()))
    return {name: _as_numpy(value, backend) for name, value in values.items()}, state


def _reference_step(
    model: Any,
    kind: ModelKind,
    *,
    glimpse: torch.Tensor,
    state: Any,
    viewpoint: TorchViewpoint,
) -> tuple[dict[str, np.ndarray], Any]:
    with torch.no_grad():
        result = model(glimpse=glimpse, state=state, viewpoint=viewpoint)
    values, state = _split_result(result, kind)
    if kind == "pretraining":
        values |= _teacher_values(model, state)
    return {name: value.detach().cpu().numpy() for name, value in values.items()}, state


def _metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise AssertionError(f"output shape mismatch: reference={reference.shape}, native={candidate.shape}")
    difference = candidate - reference
    reference_norm = float(np.linalg.norm(reference.ravel()))
    difference_norm = float(np.linalg.norm(difference.ravel()))
    scale = max(reference_norm, np.finfo(np.float64).eps)
    if reference.ndim >= 2:
        reference_mean = reference.mean(axis=-1, keepdims=True)
        candidate_mean = candidate.mean(axis=-1, keepdims=True)
        reference_std = np.sqrt(((reference - reference_mean) ** 2).mean(axis=-1, keepdims=True) + 1e-5)
        candidate_std = np.sqrt(((candidate - candidate_mean) ** 2).mean(axis=-1, keepdims=True) + 1e-5)
        normalized_reference = (reference - reference_mean) / reference_std
        normalized_candidate = (candidate - candidate_mean) / candidate_std
    else:
        normalized_reference = reference / scale
        normalized_candidate = candidate / max(float(np.linalg.norm(candidate.ravel())), np.finfo(np.float64).eps)
    normalized_difference = normalized_candidate - normalized_reference
    return {
        "max_abs": float(np.max(np.abs(difference), initial=0.0)),
        "relative_l2": float(difference_norm / scale),
        "normalized_l2": float(np.linalg.norm(normalized_difference.ravel())),
        "normalized_rmse": float(np.sqrt(np.mean(normalized_difference**2))),
    }


def verify_outputs(
    torch_model: Any,
    native_model: Any,
    module: Any,
    *,
    kind: ModelKind,
    backend: BackendName,
    canvas_grid_size: int,
    glimpse_size_px: int,
    steps: int,
    image: Path | None,
    scene_size_px: int,
) -> dict[str, dict[str, float]]:
    if canvas_grid_size < 1 or glimpse_size_px < 1 or steps < 1:
        raise ValueError(f"verification dimensions must be positive: {canvas_grid_size=}, {glimpse_size_px=}, {steps=}")
    if image is None:
        generator = torch.Generator().manual_seed(9017)
        scene = torch.randn((1, 79, 61, 3), generator=generator, dtype=torch.float32)
        scene_nhwc = scene.numpy()
        scene_nchw = scene.permute(0, 3, 1, 2)
    else:
        if not image.is_file():
            raise FileNotFoundError(f"verification image does not exist: {image}")
        with Image.open(image) as source_image:
            pil_image = source_image.convert("RGB")
        scene_nhwc = preprocess(scene_size_px)(pil_image)[None]
        scene_nchw = cast(torch.Tensor, torch_preprocess(scene_size_px)(pil_image)).unsqueeze(0)
    centers = np.asarray([[0.0, 0.0], [-0.21, 0.27], [0.42, -0.31]], dtype=np.float32)
    scales = np.asarray([0.82, 0.48, 0.37], dtype=np.float32)
    torch_state = torch_model.init_state(batch_size=1, canvas_grid_size=canvas_grid_size)
    native_state = native_model.init_state(batch_size=1, canvas_grid_size=canvas_grid_size)
    metrics: dict[str, dict[str, float]] = {}
    for step in range(steps):
        center = centers[step % len(centers)][None]
        scale = scales[step % len(scales)][None]
        torch_viewpoint = TorchViewpoint(centers=torch.from_numpy(center), scales=torch.from_numpy(scale))
        native_viewpoint = _native_viewpoint(module, backend, center, scale)
        torch_glimpse = torch_sample_at_viewpoint(
            spatial=scene_nchw,
            viewpoint=torch_viewpoint,
            glimpse_size_px=glimpse_size_px,
        )
        native_scene = _native_array(scene_nhwc, backend)
        native_glimpse = module.sample_at_viewpoint(
            spatial=native_scene,
            viewpoint=native_viewpoint,
            glimpse_size_px=glimpse_size_px,
        )
        reference, torch_state = _reference_step(
            torch_model,
            kind,
            glimpse=torch_glimpse,
            state=torch_state,
            viewpoint=torch_viewpoint,
        )
        native_values, native_state = _native_step(
            native_model,
            kind,
            glimpse=native_glimpse,
            state=native_state,
            viewpoint=native_viewpoint,
            backend=backend,
        )
        for name, value in reference.items():
            if not np.isfinite(value).all() or not np.isfinite(native_values[name]).all():
                raise FloatingPointError(f"non-finite {name} at verification step {step}")
            metrics[f"step_{step}.{name}"] = _metric(value, native_values[name])
    return metrics


def _assert_metrics(metrics: dict[str, dict[str, float]]) -> None:
    non_finite = {
        name: values for name, values in metrics.items()
        if not all(np.isfinite(value) for value in values.values())
    }
    if non_finite:
        raise FloatingPointError(f"non-finite checkpoint parity metrics: {non_finite}")
    failures = {
        name: values
        for name, values in metrics.items()
        if (
            values["relative_l2"] > 5e-3 and values["max_abs"] > 5e-4
        ) or values["normalized_rmse"] > 5e-4
    }
    if failures:
        raise AssertionError(f"native checkpoint output parity failed: {failures}")


def run(config: ConvertCheckpoints) -> None:
    if config.output.exists():
        raise FileExistsError(f"refusing to overwrite existing checkpoint directory: {config.output}")
    source, source_revision = _source_snapshot(config.source, config.revision)
    record = _read_config(source)
    kind = _model_kind(record, config.kind)
    if kind == "probe":
        _run_probe_conversion(config, source=source, source_revision=source_revision, record=record)
        return
    model_config = record
    torch_model = _load_torch_model(source, kind)
    module = _native_module(config.backend)
    native_model = _native_model(module, config.backend, kind, model_config)
    _load_converted_weights(torch_model, native_model, config.backend)
    patch_size = native_model.canvit.patch_size
    glimpse_size_px = config.glimpse_size_px
    if glimpse_size_px % patch_size:
        raise ValueError(f"glimpse_size_px={glimpse_size_px} is not divisible by patch_size={patch_size}")
    metrics_before_save = verify_outputs(
        torch_model,
        native_model,
        module,
        kind=kind,
        backend=config.backend,
        canvas_grid_size=config.canvas_grid_size,
        glimpse_size_px=glimpse_size_px,
        steps=config.steps,
        image=config.image,
        scene_size_px=config.scene_size_px,
    )
    _assert_metrics(metrics_before_save)
    for name, values in metrics_before_save.items():
        log.info("%s max_abs=%.3e relative_l2=%.3e normalized_rmse=%.3e", name, values["max_abs"],
                 values["relative_l2"], values["normalized_rmse"])
    native_model.save_pretrained(config.output)
    reloaded = type(native_model).from_pretrained(config.output)
    metrics_after_reload = verify_outputs(
        torch_model,
        reloaded,
        module,
        kind=kind,
        backend=config.backend,
        canvas_grid_size=config.canvas_grid_size,
        glimpse_size_px=glimpse_size_px,
        steps=config.steps,
        image=config.image,
        scene_size_px=config.scene_size_px,
    )
    _assert_metrics(metrics_after_reload)
    (config.output / "conversion.json").write_text(json.dumps({
        "source": config.source,
        "source_revision": source_revision,
        "backend": config.backend,
        "kind": kind,
        "canvas_grid_size": config.canvas_grid_size,
        "glimpse_size_px": glimpse_size_px,
        "steps": config.steps,
        "scene_size_px": config.scene_size_px,
        "verification_image": str(config.image) if config.image is not None else None,
        "verification_image_sha256": (
            hashlib.sha256(config.image.read_bytes()).hexdigest() if config.image is not None else None
        ),
        "metrics_before_save": metrics_before_save,
        "metrics_after_reload": metrics_after_reload,
    }, indent=2, sort_keys=True) + "\n")
    log.info("wrote %s checkpoint to %s", config.backend, config.output)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(tyro.cli(ConvertCheckpoints))


if __name__ == "__main__":
    main()
