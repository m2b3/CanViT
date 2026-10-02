from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from canvit_pytorch import CanViT as TorchCanViT
from canvit_pytorch import Viewpoint as TorchViewpoint
from canvit_pytorch import sample_at_viewpoint as torch_sample_at_viewpoint

from tools.checkpoint_conversion import (
    BackendName,
    convert_named_arrays,
    load_native_arrays,
    native_state_arrays,
)


@dataclass(frozen=True)
class Backend:
    name: BackendName
    module: Any

    def array(self, value: np.ndarray) -> Any:
        if self.name == "mlx":
            import mlx.core as mx

            return mx.array(value)
        import jax.numpy as jnp

        return jnp.asarray(value)

    def to_numpy(self, value: Any) -> np.ndarray:
        if self.name == "mlx":
            import mlx.core as mx

            mx.eval(value)
        return np.asarray(value)

    def make_canvit(self, config: Any) -> Any:
        if self.name == "mlx":
            return self.module.CanViT(config)
        from flax import nnx

        return self.module.CanViT(config, rngs=nnx.Rngs(0))

    def make_classifier(self, config: Any, n_classes: int) -> Any:
        kwargs = {"canvit_config": config, "n_classes": n_classes}
        if self.name == "nnx":
            from flax import nnx

            kwargs["rngs"] = nnx.Rngs(0)
        return self.module.CanViTForImageClassification(**kwargs)

    def make_pretraining(self, config: Any, teacher_dim: int, teacher_patch_grid: int) -> Any:
        kwargs = {"canvit_config": config, "teacher_dim": teacher_dim, "teacher_patch_grid": teacher_patch_grid}
        if self.name == "nnx":
            from flax import nnx

            kwargs["rngs"] = nnx.Rngs(0)
        return self.module.CanViTForPretraining(**kwargs)

    def make_probe(self, *, embed_dim: int, num_classes: int, dropout: float, use_ln: bool) -> Any:
        kwargs: dict[str, Any] = {
            "embed_dim": embed_dim,
            "num_classes": num_classes,
            "dropout": dropout,
            "use_ln": use_ln,
        }
        if self.name == "nnx":
            from flax import nnx

            kwargs["rngs"] = nnx.Rngs(0)
        return self.module.SegmentationProbe(**kwargs)

    def set_probe_eval(self, probe: Any) -> None:
        probe.eval()
        if self.name == "nnx":
            probe.bn.use_running_average = True

    def set_probe_train(self, probe: Any) -> None:
        probe.train()
        if self.name == "nnx":
            probe.bn.use_running_average = False

    def reference_pair(self, config: Any) -> tuple[TorchCanViT, Any]:
        torch.manual_seed(0)
        reference = TorchCanViT(config).eval()
        native = self.make_canvit(config)
        source = {name: value.detach().cpu().numpy() for name, value in reference.state_dict().items()}
        schema = {name: tuple(value.shape) for name, value in native_state_arrays(native, backend=self.name).items()}
        converted = convert_named_arrays(source, target_shapes=schema, backend=self.name)
        load_native_arrays(native, converted, backend=self.name)
        return reference, native

    def load_reference_weights(self, native: Any, reference: torch.nn.Module) -> None:
        source = {name: value.detach().cpu().numpy() for name, value in reference.state_dict().items()}
        schema = {name: tuple(value.shape) for name, value in native_state_arrays(native, backend=self.name).items()}
        converted = convert_named_arrays(source, target_shapes=schema, backend=self.name)
        load_native_arrays(native, converted, backend=self.name)

    def viewpoint(self, centers: np.ndarray, scales: np.ndarray) -> Any:
        return self.module.Viewpoint(centers=self.array(centers), scales=self.array(scales))

    def inputs(
        self, *, batch_size: int = 2, glimpse_size_px: int = 8,
    ) -> tuple[torch.Tensor, TorchViewpoint, Any, Any]:
        generator = torch.Generator().manual_seed(1234)
        scene = torch.randn((batch_size, 3, 23, 17), generator=generator)
        centers = np.asarray([[0.0, 0.0], [-0.34, 0.22]], dtype=np.float32)[:batch_size]
        scales = np.asarray([0.79, 0.46], dtype=np.float32)[:batch_size]
        torch_viewpoint = TorchViewpoint(torch.from_numpy(centers), torch.from_numpy(scales))
        torch_glimpse = torch_sample_at_viewpoint(
            spatial=scene, viewpoint=torch_viewpoint, glimpse_size_px=glimpse_size_px,
        )
        native_viewpoint = self.viewpoint(centers, scales)
        native_scene = self.array(scene.permute(0, 2, 3, 1).numpy())
        native_glimpse = self.module.sample_at_viewpoint(
            spatial=native_scene, viewpoint=native_viewpoint, glimpse_size_px=glimpse_size_px,
        )
        return torch_glimpse, torch_viewpoint, native_glimpse, native_viewpoint


def canvit_values(output: Any, backend: Backend) -> dict[str, np.ndarray]:
    values = {
        "canvas": output.state.canvas,
        "recurrent_cls": output.state.recurrent_cls,
        "glimpse_patches": output.glimpse_patches,
    }
    if output.vpe is not None:
        values["vpe"] = output.vpe
    return {name: backend.to_numpy(array) for name, array in values.items()}


def classifier_values(result: tuple[Any, Any], backend: Backend) -> dict[str, np.ndarray]:
    logits, state = result
    values = {"logits": logits, "canvas": state.canvas, "recurrent_cls": state.recurrent_cls}
    return {name: backend.to_numpy(array) for name, array in values.items()}


def relative_l2(reference: np.ndarray, actual: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    return float(np.linalg.norm((actual - reference).ravel()) / max(np.linalg.norm(reference.ravel()), 1e-12))


def normalized_l2(reference: np.ndarray, actual: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64).ravel()
    actual = np.asarray(actual, dtype=np.float64).ravel()
    reference /= max(np.linalg.norm(reference), 1e-12)
    actual /= max(np.linalg.norm(actual), 1e-12)
    return float(np.linalg.norm(actual - reference))


def tokenwise_normalized_l2(reference: np.ndarray, actual: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    reference_mean = reference.mean(axis=-1, keepdims=True)
    actual_mean = actual.mean(axis=-1, keepdims=True)
    reference_std = np.sqrt(((reference - reference_mean) ** 2).mean(axis=-1, keepdims=True) + 1e-5)
    actual_std = np.sqrt(((actual - actual_mean) ** 2).mean(axis=-1, keepdims=True) + 1e-5)
    return float(np.linalg.norm(
        ((actual - actual_mean) / actual_std - (reference - reference_mean) / reference_std).ravel(),
    ))
