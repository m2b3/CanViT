"""The analytic FLOP counts reproduce the paper's, and torch's FLOP counter measures the same on real forwards."""

from collections.abc import Callable
from dataclasses import replace
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import NamedTuple

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode
from transformers import AutoConfig, DINOv3ViTConfig, DINOv3ViTModel

from canvit_pytorch import CanViT, CanViTConfig, CanViTForSemanticSegmentation, SegmentationProbe
from canvit_pytorch.benchmarks.ade20k import NUM_CLASSES
from canvit_pytorch.flops import (
    attention_flops,
    canvas_attention_read_flops,
    canvas_attention_write_flops,
    dinov3_flops,
    glimpse_flops,
    linear_flops,
    num_canvas_tokens,
    num_glimpse_tokens,
    segmentation_probe_flops,
)
from canvit_pytorch.pretrain.ablations import ABLATIONS, AblationSlug
from canvit_pytorch.pretrain.config import PretrainingConfig
from canvit_pytorch.teacher import DINOV3_REPOS, DINOv3Variant
from tests.model.tiny import GLIMPSE_PX, TINY, glimpse_batch

CANVIT_B = CanViTConfig()
PAPER_GLIMPSE_PX = 128

# Reference values: the paper's FLOP exporter (m2b3/CanViT-paper-exporter @ 49966e5,
# src/canvit_paper_exporter/flops/), run unmodified on 2026-09-26.


class ExporterCanViTB(NamedTuple):
    glimpse: int  # canvit.py::canvit_forward_flops_per_glimpse: backbone + canvas_read + canvas_write
    ade20k_probe: int  # its seg_head
    read: int  # canvit.py::canvas_read_flops(..., full_proj=False)
    write: int  # canvit.py::canvas_write_flops(..., full_proj=False, convex=False)
    read_qkvo: int  # full_proj=True
    write_qkvo: int


EXPORTER_CANVIT_B = {  # by canvas grid G: the exporter's arch.py::CANVIT_B with canvas_grid=G
    8: ExporterCanViTB(13_801_724_160, 19_660_800, 246_611_968, 246_611_968, 582_156_288, 582_156_288),
    16: ExporterCanViTB(14_136_744_192, 78_643_200, 302_448_640, 302_448_640, 1_443_299_328, 1_443_299_328),
    32: ExporterCanViTB(15_476_824_320, 314_572_800, 525_795_328, 525_795_328, 4_887_871_488, 4_887_871_488),
    64: ExporterCanViTB(20_837_144_832, 1_258_291_200, 1_419_182_080, 1_419_182_080, 18_666_160_128, 18_666_160_128),
    128: ExporterCanViTB(42_278_426_880, 5_033_164_800, 4_992_729_088, 4_992_729_088, 73_779_314_688, 73_779_314_688),
}
EXPORTER_DINOV3: dict[DINOv3Variant, dict[int, int]] = {  # teacher.py::teacher_flops, by input size in px
    "vits16": {128: 3_055_749_120, 144: 3_836_289_024, 160: 4_721_264_640, 192: 6_821_775_360, 256: 12_490_573_824,
               384: 31_235_180_544, 512: 63_819_417_600},
    "vitb16": {128: 11_971_989_504, 144: 14_976_958_464, 160: 18_360_668_160, 192: 26_298_814_464,
               256: 47_149_092_864, 384: 111_817_396_224, 512: 215_036_596_224},
}
# ablations/static.py: each variant's canvit_forward_flops_per_glimpse(...).total, ADE20K probe included,
# behind the GFLOPs column of the paper's ablation table.
EXPORTER_ABLATIONS: dict[AblationSlug, int] = {
    "baseline": 15_791_397_120, "dcan256": 13_189_388_544, "qkvo-dcan256": 14_825_167_104,
    "qkvo-dcan384": 17_303_558_400, "no-reads": 14_214_011_136, "rw-stride6": 13_688_215_808,
    "no-dense": 15_791_397_120, "no-fiid-1riid": 15_791_397_120, "no-fiid-2riid": 15_791_397_120,
    "no-bptt": 15_791_397_120, "vit-s": 5_945_150_592, "no-vpe": 15_571_894_272,
}


def ade20k_glimpse_flops(config: CanViTConfig, *, canvas_grid_size: int) -> int:
    """What the paper reports for ADE20K: a 128 px glimpse, then the linear probe on the canvas."""
    return glimpse_flops(config, glimpse_size_px=PAPER_GLIMPSE_PX, canvas_grid_size=canvas_grid_size) + (
        segmentation_probe_flops(grid_size=canvas_grid_size, embed_dim=config.canvas_dim, num_classes=NUM_CLASSES)
    )


def read_write_pair_flops(config: CanViTConfig, *, canvas_grid_size: int) -> int:
    geometry = {"glimpse_size_px": PAPER_GLIMPSE_PX, "canvas_grid_size": canvas_grid_size}
    return canvas_attention_read_flops(config, **geometry) + canvas_attention_write_flops(config, **geometry)


def one_decimal(value: Decimal) -> str:
    """The paper's rounding: half up."""
    return str(value.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def gflops(flops: int) -> str:
    return one_decimal(Decimal(flops) / 10**9)


@pytest.mark.parametrize("canvas_grid_size", EXPORTER_CANVIT_B)
def test_canvit_b_matches_the_paper_exporter(canvas_grid_size: int):
    expected = EXPORTER_CANVIT_B[canvas_grid_size]
    geometry = {"glimpse_size_px": PAPER_GLIMPSE_PX, "canvas_grid_size": canvas_grid_size}
    qkvo = replace(CANVIT_B, canvas_projections="qkvo")
    assert glimpse_flops(CANVIT_B, **geometry) == expected.glimpse
    assert segmentation_probe_flops(
        grid_size=canvas_grid_size, embed_dim=CANVIT_B.canvas_dim, num_classes=NUM_CLASSES,
    ) == expected.ade20k_probe
    assert canvas_attention_read_flops(CANVIT_B, **geometry) == expected.read
    assert canvas_attention_write_flops(CANVIT_B, **geometry) == expected.write
    assert canvas_attention_read_flops(qkvo, **geometry) == expected.read_qkvo
    assert canvas_attention_write_flops(qkvo, **geometry) == expected.write_qkvo


@pytest.mark.parametrize("slug", list(ABLATIONS))
def test_each_ablation_matches_the_paper_exporter(slug: AblationSlug):
    base = PretrainingConfig(shards_dir=Path("shards"), images_dir=Path("images"), validation_dir=Path("val"),
                             dataset="in21k", checkpoints_dir=Path("checkpoints"), run_name="run")
    config = ABLATIONS[slug].configure(base)
    assert (config.glimpse_size_px, config.canvas_grid_size) == (PAPER_GLIMPSE_PX, 32)
    assert ade20k_glimpse_flops(config.model, canvas_grid_size=config.canvas_grid_size) == EXPORTER_ABLATIONS[slug]


def test_the_numbers_printed_in_the_paper():
    qkvo = replace(CANVIT_B, canvas_projections="qkvo")
    num_glimpse = num_glimpse_tokens(CANVIT_B, glimpse_size_px=PAPER_GLIMPSE_PX)
    num_canvas = num_canvas_tokens(CANVIT_B, canvas_grid_size=32)
    canvas_dim = CANVIT_B.canvas_dim
    # Section 4: N_g = 71, and one canvas-side projection costs D_can / (2 N_g) = 7.2 times the attention it joins.
    assert num_glimpse == 71
    projection = linear_flops(num_tokens=num_canvas, in_dim=canvas_dim, out_dim=canvas_dim)
    attention = attention_flops(num_queries=num_glimpse, num_keys=num_canvas, dim=canvas_dim)
    assert one_decimal(Decimal(projection) / attention) == "7.2"
    # Section 4: with a 64² canvas, canvas-side QKVO projections would take a Read/Write pair from 2.8 to 37.3 GFLOPs.
    assert gflops(read_write_pair_flops(CANVIT_B, canvas_grid_size=64)) == "2.8"
    assert gflops(read_write_pair_flops(qkvo, canvas_grid_size=64)) == "37.3"
    # The passive-vision comparison table: one full-scene glimpse onto a 32² canvas.
    assert gflops(ade20k_glimpse_flops(CANVIT_B, canvas_grid_size=32)) == "15.8"


@pytest.mark.network
@pytest.mark.parametrize("variant", list(EXPORTER_DINOV3))
def test_dinov3_from_its_hub_config_matches_the_paper_exporter(variant: DINOv3Variant):
    config = AutoConfig.from_pretrained(DINOV3_REPOS[variant])  # gated: a token that accepted DINOv3's license
    assert isinstance(config, DINOv3ViTConfig)
    for input_size_px, expected in EXPORTER_DINOV3[variant].items():
        assert dinov3_flops(config, input_size_px=input_size_px) == expected
    if variant == "vitb16":
        # The passive-vision comparison table: the FLOP-matched teacher sees 160 px.
        assert gflops(dinov3_flops(config, input_size_px=160)) == "18.4"


def measured_flops(forward: Callable[[], object]) -> dict[str, int]:
    """FLOPs of one call by module path ("Global" for the total), as torch's FLOP counter traces them."""
    # The counter misses the fused CPU attention kernel (torch 2.14: zero FLOPs); the math backend runs bmm.
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as counter:
        forward()
    return {path: sum(by_operator.values()) for path, by_operator in counter.get_flop_counts().items()}


def test_the_flop_counter_measures_attention():
    queries, keys, values = torch.randn(2, 3, 5, 8), torch.randn(2, 3, 7, 8), torch.randn(2, 3, 7, 8)
    measured = measured_flops(lambda: F.scaled_dot_product_attention(queries, keys, values))
    # Batch 2, 3 heads, 5 queries, 7 keys, 8 dims per head: scores, then the weighted sum of values.
    assert measured["Global"] == 2 * (2 * 2 * 3 * 5 * 7 * 8)


@pytest.mark.parametrize("config", [
    TINY, replace(TINY, canvas_projections="qkvo"), replace(TINY, enable_reads=False),
    replace(TINY, enable_vpe=False), replace(TINY, rw_stride=6),
], ids=["asymmetric", "qkvo", "no-reads", "no-vpe", "rw-stride6"])
def test_canvit_counts_match_the_flop_counter(config: CanViTConfig):
    torch.manual_seed(0)
    probe = SegmentationProbe(embed_dim=config.canvas_dim, num_classes=NUM_CLASSES, dropout=0.0, use_ln=True)
    model = CanViTForSemanticSegmentation(canvit=CanViT(config), probe=probe).eval()
    batch_size, canvas_grid_size = 2, 4
    glimpse, viewpoint = glimpse_batch(seed=0, batch_size=batch_size)
    state = model.init_state(batch_size=batch_size, canvas_grid_size=canvas_grid_size)
    measured = measured_flops(lambda: model(glimpse=glimpse, state=state, viewpoint=viewpoint))
    geometry = {"glimpse_size_px": GLIMPSE_PX, "canvas_grid_size": canvas_grid_size}
    root = type(model).__name__
    assert measured[f"{root}.canvit"] == batch_size * glimpse_flops(config, **geometry)
    read, write = canvas_attention_read_flops(config, **geometry), canvas_attention_write_flops(config, **geometry)
    for i in range(len(model.canvit.canvas_reads)):
        assert measured[f"{root}.canvit.canvas_reads.{i}"] == batch_size * read
    for i in range(len(model.canvit.canvas_writes)):
        assert measured[f"{root}.canvit.canvas_writes.{i}"] == batch_size * write
    assert measured[f"{root}.probe"] == batch_size * segmentation_probe_flops(
        grid_size=canvas_grid_size, embed_dim=config.canvas_dim, num_classes=NUM_CLASSES,
    )
    assert measured["Global"] == measured[f"{root}.canvit"] + measured[f"{root}.probe"]


def test_dinov3_count_matches_the_flop_counter():
    config = DINOv3ViTConfig(hidden_size=64, num_attention_heads=2, intermediate_size=192, num_hidden_layers=2,
                             num_register_tokens=4, patch_size=16)
    model, image = DINOv3ViTModel(config).eval(), torch.randn(1, 3, 48, 48)
    assert measured_flops(lambda: model(image))["Global"] == dinov3_flops(config, input_size_px=48)
