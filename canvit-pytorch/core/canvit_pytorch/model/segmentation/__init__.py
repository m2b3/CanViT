"""CanViT + SegmentationProbe head, in one ``nn.Module``.

Construction goes through :meth:`CanViTForSemanticSegmentation.from_pretrained_with_probe`,
which loads a pretrained CanViT and a separately trained ``SegmentationProbe``
from repo IDs or local checkpoint directories so callers do not manage the two halves
separately. Mirrors :class:`CanViTForImageClassification`'s API shape.
"""

import logging
from typing import cast, get_args

from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn
from torch.nn import functional as F

from canvit_pytorch.backbone import BackboneName, create_backbone
from canvit_pytorch.model.base.config import CanViTConfig
from canvit_pytorch.model.base.impl import CanViT, RecurrentState
from canvit_pytorch.model.hub_mixin import SafeHubMixin
from canvit_pytorch.model.pretraining.hub import CanViTForPretrainingHFHub
from canvit_pytorch.probes import SegmentationProbe
from canvit_pytorch.viewpoint import Viewpoint

log = logging.getLogger(__name__)


class CanViTForSemanticSegmentation(
    nn.Module,
    SafeHubMixin,
    PyTorchModelHubMixin,
    library_name="canvit-pytorch",
    repo_url="https://github.com/m2b3/CanViT-PyTorch",
):
    """:class:`CanViT` (``self.canvit``) + :class:`SegmentationProbe` (``self.head``).

    The wrapped CanViT is bare — no pretraining heads or standardizers,
    since those are unused for downstream segmentation.

    Example::

        seg = CanViTForSemanticSegmentation.from_pretrained_with_probe(
            pretrained_repo="<org>/canvitb16-add-vpe-pretrain-...",
            probe_repo="<org>/probe-ade20k-40k-s512-c32-in21k",
        ).eval()

        state = seg.init_state(batch_size=B, canvas_grid_size=32)
        logits, state = seg(glimpse=glimpse, state=state, viewpoint=vp)
        # logits: [B, num_classes, 32, 32]
    """

    def __init__(
        self,
        *,
        backbone_name: BackboneName,
        model_config: dict,
        num_classes: int,
        dropout: float = 0.1,
        use_ln: bool = True,
    ):
        super().__init__()
        # HF config.json may carry pretraining-only fields (e.g. teacher_dim)
        # that CanViTConfig doesn't accept; filter to known fields.
        known_fields = CanViTConfig.__dataclass_fields__
        cfg_dict = {k: v for k, v in model_config.items() if k in known_fields}
        cfg = CanViTConfig(**cfg_dict)
        self.canvit = CanViT(backbone=create_backbone(backbone_name), cfg=cfg)
        # Head consumes canvas spatial tokens, so its dim is canvas_dim (not local_dim).
        D = self.canvit.canvas_dim
        self.head = SegmentationProbe(
            embed_dim=D,
            num_classes=num_classes,
            dropout=dropout,
            use_ln=use_ln,
        )

    @property
    def canvas_dim(self) -> int:
        return self.canvit.canvas_dim

    @property
    def num_classes(self) -> int:
        return self.head.num_classes

    def init_state(self, *, batch_size: int, canvas_grid_size: int) -> RecurrentState:
        return self.canvit.init_state(batch_size=batch_size, canvas_grid_size=canvas_grid_size)

    def forward(
        self, *, glimpse: Tensor, state: RecurrentState, viewpoint: Viewpoint,
    ) -> tuple[Tensor, RecurrentState]:
        """One CanViT step + head application.

        Returns ``(logits [B, num_classes, G, G], new_state)`` where ``G`` is the
        canvas grid size of ``state``. Use :meth:`predict` to also bilinearly
        upsample logits to a target spatial resolution.

        For CanViT-only execution without the segmentation head, call ``self.canvit(...)`` directly.
        For head-only on a cached state, call ``self.head(spatial_hwd)`` directly.
        """
        out = self.canvit(glimpse=glimpse, state=state, viewpoint=viewpoint)
        spatial = self.canvit.get_spatial(out.state.canvas)  # [B, G*G, D]
        B, n_spatial, D = spatial.shape
        canvas_grid = int(n_spatial ** 0.5)
        assert canvas_grid * canvas_grid == n_spatial, (
            f"Canvas has {n_spatial} spatial tokens, not a perfect square — "
            f"init_state must be called with a valid canvas_grid_size."
        )
        return self.head(spatial.view(B, canvas_grid, canvas_grid, D)), out.state

    def predict(
        self,
        *,
        glimpse: Tensor,
        state: RecurrentState,
        viewpoint: Viewpoint,
        target_size: tuple[int, int],
    ) -> tuple[Tensor, RecurrentState]:
        """:meth:`forward` + bilinear upsample of logits to ``target_size``.

        Returns ``(logits [B, num_classes, *target_size], new_state)``.
        """
        logits, new_state = self(glimpse=glimpse, state=state, viewpoint=viewpoint)
        return F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False), new_state

    @classmethod
    def from_pretrained_with_probe(
        cls,
        *,
        pretrained_repo: str,
        probe_repo: str,
    ) -> "CanViTForSemanticSegmentation":
        """Load a pretrained CanViT + a published seg probe; bundle them into one model.

        Pretraining-only modules (``scene_cls_head``, ``scene_patches_head``,
        ``cls_standardizers``, ``scene_standardizers``) on the loaded model
        are discarded.
        """
        log.info("Loading pretrained CanViT from %s", pretrained_repo)
        pretrained = CanViTForPretrainingHFHub.from_pretrained(pretrained_repo)
        D = pretrained.canvas_dim

        log.info("Loading probe from %s", probe_repo)
        probe = SegmentationProbe.from_pretrained(probe_repo)
        assert probe.embed_dim == D, (
            f"Probe expects embed_dim={probe.embed_dim} but the CanViT produces "
            f"canvas_dim={D}. Probe was trained for a different model variant."
        )

        cfg = pretrained.cfg
        assert pretrained.backbone_name in get_args(BackboneName), (
            f"Unknown ViT backbone: {pretrained.backbone_name!r}"
        )
        model = cls(
            backbone_name=cast(BackboneName, pretrained.backbone_name),
            model_config={k: v for k, v in vars(cfg).items() if k in CanViTConfig.__dataclass_fields__},
            num_classes=probe.num_classes,
            dropout=probe.dropout_p,
            use_ln=probe.use_ln,
        )

        # Copy bare-CanViT weights (drop pretraining-only modules)
        pretraining_only_prefixes = (
            "scene_cls_head.", "scene_patches_head.",
            "cls_standardizers.", "scene_standardizers.",
        )
        base_sd = {
            k: v for k, v in pretrained.state_dict().items()
            if not any(k.startswith(p) for p in pretraining_only_prefixes)
        }
        missing, unexpected = model.canvit.load_state_dict(base_sd, strict=False)
        assert not missing, f"Missing CanViT keys: {missing}"
        assert not unexpected, f"Unexpected CanViT keys: {unexpected}"

        # Copy seg head weights
        missing, unexpected = model.head.load_state_dict(probe.state_dict(), strict=True)
        assert not missing and not unexpected, (
            f"Probe state_dict mismatch: missing={missing}, unexpected={unexpected}"
        )

        log.info(
            "Constructed CanViTForSemanticSegmentation: %d classes, canvas_dim=%d, dropout=%s, use_ln=%s",
            probe.num_classes, D, probe.dropout_p, probe.use_ln,
        )
        return model
