"""CanViT with a linear classifier on the recurrent CLS token."""

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from canvit_core.checkpoint import hub_file
from canvit_core.config import CanViTConfig
from canvit_core.readout import fuse_probe
from safetensors.numpy import load_file

from canvit_nnx.hub import HubMixin
from canvit_nnx.model.canvit import CanViT, RecurrentState
from canvit_nnx.model.pretraining import CanViTForPretraining
from canvit_nnx.model.readout import LinearReadout
from canvit_nnx.viewpoint import Viewpoint

Array = jax.Array


class CanViTForImageClassification(nnx.Module, HubMixin):
    def __init__(
        self,
        *,
        canvit_config: CanViTConfig,
        n_classes: int,
        rngs: nnx.Rngs,
    ) -> None:
        assert n_classes > 0, n_classes
        self.canvit = CanViT(canvit_config, rngs=rngs)
        self.readout = LinearReadout(
            in_dim=self.canvit.backbone_dim,
            out_dim=n_classes,
            rngs=rngs,
        )

    @property
    def n_classes(self) -> int:
        return self.readout.proj.out_features

    def init_state(self, *, batch_size: int, canvas_grid_size: int) -> RecurrentState:
        return self.canvit.init_state(batch_size=batch_size, canvas_grid_size=canvas_grid_size)

    def __call__(self, *, glimpse: Array, state: RecurrentState, viewpoint: Viewpoint) -> tuple[Array, RecurrentState]:
        output = self.canvit(glimpse=glimpse, state=state, viewpoint=viewpoint)
        return self.classify(output.state.recurrent_cls), output.state

    def classify(self, recurrent_cls: Array) -> Array:
        assert recurrent_cls.ndim == 3 and recurrent_cls.shape[1] == 1, recurrent_cls.shape
        return self.readout(recurrent_cls[:, 0, :].astype(jnp.float32))

    def checkpoint_config(self) -> dict[str, object]:
        return {
            "canvit_config": dataclasses.asdict(self.canvit.config),
            "n_classes": self.n_classes,
        }

    @classmethod
    def from_pretrained_with_probe(
        cls,
        *,
        pretrained_repo: str,
        probe_repo: str,
    ) -> "CanViTForImageClassification":
        pretrained = CanViTForPretraining.from_pretrained(pretrained_repo)
        probe = load_file(hub_file(probe_repo, "model.safetensors"))
        projection = pretrained.teacher_cls_readout.proj
        assert projection.bias is not None
        weight, bias = fuse_probe(
            proj_weight=np.asarray(projection.kernel[...]).T,
            proj_bias=np.asarray(projection.bias[...]),
            mean=np.asarray(pretrained.teacher_cls_standardizer.mean[0]),
            std=np.asarray(pretrained.teacher_cls_standardizer.std[0]),
            probe_weight=probe["weight"],
            probe_bias=probe["bias"],
        )
        model = cls(
            canvit_config=pretrained.canvit.config,
            n_classes=weight.shape[0],
            rngs=nnx.Rngs(0),
        )
        model.canvit = pretrained.canvit
        model.readout.norm = pretrained.teacher_cls_readout.norm
        model.readout.proj.kernel[...] = jnp.asarray(weight).T
        assert model.readout.proj.bias is not None
        model.readout.proj.bias[...] = jnp.asarray(bias)
        return model
