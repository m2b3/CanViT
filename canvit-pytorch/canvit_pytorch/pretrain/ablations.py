"""The paper's pretraining ablations (Appendix E, Table 5): short runs that each change one design choice.

Every ablation, the baseline included, trains 214,656 steps (43 jobs of 4,992; the paper's
"approximately 215k") with a 20k-step warmup:

    python -m canvit_pytorch.pretrain --ablation no-reads ...
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal, get_args

from canvit_pytorch.hub.repos import hub_repo
from canvit_pytorch.pretrain.config import PretrainingConfig

ABLATION_TOTAL_STEPS = 214_656
ABLATION_WARMUP_STEPS = 20_000

AblationSlug = Literal[
    "baseline", "dcan256", "qkvo-dcan256", "qkvo-dcan384", "no-reads", "rw-stride6", "no-dense",
    "no-fiid-1riid", "no-fiid-2riid", "no-bptt", "vit-s", "no-vpe",
]

Change = Callable[[PretrainingConfig], PretrainingConfig]


def _model(**fields: Any) -> Change:
    return lambda config: replace(config, model=replace(config.model, **fields))


def _training(**fields: Any) -> Change:
    return lambda config: replace(config, **fields)


@dataclass(frozen=True)
class Ablation:
    slug: AblationSlug
    paper_row: str
    """The row letter in the paper's table; empty for the baseline."""
    paper_label: str
    released_repo: str
    change: Change

    @property
    def probe_model_name(self) -> str:
        """How ADE20K probe names refer to this checkpoint."""
        return f"abl-{self.slug}"

    def configure(self, config: PretrainingConfig) -> PretrainingConfig:
        """This ablation's run: the ablations' shorter schedule, then this ablation's change."""
        return self.change(replace(config, total_steps=ABLATION_TOTAL_STEPS, warmup_steps=ABLATION_WARMUP_STEPS))


ABLATIONS: dict[AblationSlug, Ablation] = {a.slug: a for a in [
    Ablation("baseline", "", "CanViT-B (baseline)", hub_repo("canvitb16-abl-baseline-2026-03-02"), _training()),
    Ablation("dcan256", "a", "D_can = 256, asymmetric", hub_repo("canvitb16-abl-dcan256-2026-03-02"),
             _model(canvas_num_heads=2)),
    Ablation("qkvo-dcan256", "b", "D_can = 256, + QKVO", hub_repo("canvitb16-abl-qkvo-dcan256-2026-03-02"),
             _model(canvas_num_heads=2, canvas_projections="qkvo")),
    Ablation("qkvo-dcan384", "c", "D_can = 384, + QKVO", hub_repo("canvitb16-abl-qkvo-dcan384-2026-03-02"),
             _model(canvas_num_heads=3, canvas_projections="qkvo")),
    Ablation("no-reads", "d", "No canvas reads", hub_repo("canvitb16-abl-no-reads-2026-03-02"),
             _model(enable_reads=False)),
    Ablation("rw-stride6", "e", "RW stride = 6 (1R / 1W)", hub_repo("canvitb16-abl-rw-stride6-2026-03-03"),
             _model(rw_stride=6)),
    Ablation("no-dense", "f", "No dense supervision", hub_repo("canvitb16-abl-no-dense-2026-03-02"),
             _training(enable_teacher_patch_loss=False)),
    Ablation("no-fiid-1riid", "g", "No F-IID, 1× R-IID", hub_repo("canvitb16-abl-no-fiid-1riid-2026-03-02"),
             _training(rollout_policies=("random",))),
    Ablation("no-fiid-2riid", "h", "No F-IID, 2× R-IID", hub_repo("canvitb16-abl-no-fiid-2riid-2026-03-06"),
             _training(rollout_policies=("random", "random"))),
    # A lower stop probability keeps the mean rollout at 4 glimpses with K = 1.
    Ablation("no-bptt", "i", "No BPTT (K = 1)", hub_repo("canvitb16-abl-no-bptt-2026-03-06"),
             _training(tbptt_chunk_glimpses=1, stop_probability=0.25)),
    Ablation("vit-s", "j", "D_bb = 384 (ViT-S backbone)", hub_repo("canvitb16-abl-vit-s-2026-03-03"),
             _model(backbone_name="vits16")),
    Ablation("no-vpe", "k", "No VPE token", hub_repo("canvitb16-abl-no-vpe-2026-03-03"),
             _model(enable_vpe=False)),
]}

assert set(ABLATIONS) == set(get_args(AblationSlug)), set(ABLATIONS) ^ set(get_args(AblationSlug))
