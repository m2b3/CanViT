"""The paper's pretraining ablations as the figures and tables present them.

Each variant's model and released checkpoint come from canvit_pytorch.pretrain.ablations;
this module adds only presentation: plot label, change description and color.
"""

from dataclasses import dataclass
from pathlib import Path

from canvit_pytorch import CanViTConfig
from canvit_pytorch.pretrain.ablations import ABLATIONS, Ablation, AblationSlug
from canvit_pytorch.pretrain.config import PretrainingConfig

# Ablation changes apply to a whole pretraining run; only its model is read here.
_ANY_RUN = PretrainingConfig(
    shards_dir=Path(), images_dir=Path(), validation_dir=Path(), dataset="in21k", checkpoints_dir=Path(), run_name="",
)


@dataclass(frozen=True)
class AblationVariant:
    slug: AblationSlug
    label: str
    delta: str
    color: str

    @property
    def ablation(self) -> Ablation:
        return ABLATIONS[self.slug]

    @property
    def model_config(self) -> CanViTConfig:
        return self.ablation.change(_ANY_RUN).model


BASELINE = AblationVariant("baseline", "Baseline", "(none)", "#000000")

# Row order of the ablation tables: the paper's row letters a-k.
VARIANTS: list[AblationVariant] = [
    AblationVariant("dcan256", r"$D_{\mathrm{can}}=256$ (asym.)", "D_can 1024>256", "#2ca02c"),
    AblationVariant("qkvo-dcan256", r"+ QKVO, $D_{\mathrm{can}}=256$",
                    "canvas_proj_mode asymmetric>full, D_can 1024>256", "#1f77b4"),
    AblationVariant("qkvo-dcan384", r"+ QKVO, $D_{\mathrm{can}}=384$",
                    "canvas_proj_mode asymmetric>full, D_can 1024>384", "#17becf"),
    AblationVariant("no-reads", "No canvas reads", "enable_reads true>false", "#d62728"),
    AblationVariant("rw-stride6", "RW stride 6", "rw_stride 2>6 (3R+3W to 1R+1W at depth 12)", "#ff7f0e"),
    AblationVariant("no-dense", "No dense loss", "enable_scene_patches_loss true>false", "#8c564b"),
    AblationVariant("no-fiid-1riid", r"No F-IID (1×R)", "n_full_start_branches 1>0", "#e377c2"),
    AblationVariant("no-fiid-2riid", r"No F-IID (2×R)", "n_full_start_branches 1>0, n_random_start_branches 1>2",
                    "#c2185b"),
    AblationVariant("no-bptt", r"No BPTT ($K{=}1$)", "chunk_size 2>1, continue_prob 0.5>0.75", "#bcbd22"),
    AblationVariant("vit-s", "ViT-S backbone", "backbone_dim 768>384", "#7f7f7f"),
    AblationVariant("no-vpe", "No VPE", "enable_vpe true>false", "#9467bd"),
]

BY_SLUG: dict[str, AblationVariant] = {BASELINE.slug: BASELINE, **{v.slug: v for v in VARIANTS}}
assert [v.slug for v in VARIANTS] == sorted((a.slug for a in ABLATIONS.values() if a.paper_row), key=lambda s: ABLATIONS[s].paper_row)
