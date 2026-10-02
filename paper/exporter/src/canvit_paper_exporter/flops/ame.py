from dataclasses import dataclass

from canvit_paper_exporter.flops import primitives as P
from canvit_paper_exporter.flops.primitives import FFN_RATIO


@dataclass(frozen=True)
class AMEConfig:
    """AME (Pardyl et al. 2023, arXiv:2303.06457). ViT-L/16 encoder + 8-block decoder.

    Every value traced to a paper quote or repo line. Paper §4.1: 24 encoder
    blocks @ 1024 dim, 8 decoder blocks @ 512 dim; 256x128 input; 48px glimpses.
    Code reference: github.com/apardyl/AME:architectures/mae.py:229-234 @ ff77f27.
    """
    name: str = "AME"
    enc_dim: int = 1024
    enc_depth: int = 24
    dec_dim: int = 512
    dec_depth: int = 8
    image_h: int = 128
    image_w: int = 256
    patch_size: int = 16
    glimpse_grid: int = 3
    num_classes: int = 150

    @property
    def total_patches(self) -> int:
        return (self.image_h // self.patch_size) * (self.image_w // self.patch_size)

    @property
    def patches_per_glimpse(self) -> int:
        return self.glimpse_grid ** 2


AME = AMEConfig()


def ame_flops(num_glimpses: int, cfg: AMEConfig = AME) -> int:
    """Total AME matmul FLOPs across `num_glimpses` glimpses.

    Per `architectures/glimpse_mae.py::BaseGlimpseMae.forward` and
    `architectures/mae.py::forward_{encoder,decoder}` @ ff77f27, AME runs
    one zero-step init call followed by `num_glimpses` per-glimpse calls.
    Every call has the same shape, with the visible-patch count `n`
    growing from 0 → num_glimpses · patches_per_glimpse:

      encoder: patch_embed runs on the full image (Conv2d cannot skip
        un-sampled patches; `gather` selects visibles afterwards), then
        `enc_depth` transformer blocks over `(n + 1 CLS)` tokens.
      decoder: `decoder_embed` (Linear enc_dim → dec_dim) on those `n+1`
        tokens, then `dec_depth` blocks plus `decoder_pred` over the full
        `(total_patches + 1)` decoder sequence (CLS is sliced off AFTER
        the projection, so it pays its FLOP cost).

    To cross-check against `torch.utils.flop_counter.FlopCounterMode`, run
    the upstream forward under `sdpa_kernel([SDPBackend.MATH])` — the
    default CPU SDPA dispatcher isn't in FlopCounterMode's registry.
    """
    dec_blocks_plus_head = (
        cfg.dec_depth * P.vit_block(cfg.total_patches + 1, cfg.dec_dim, FFN_RATIO)
        + P.linear(cfg.total_patches + 1, cfg.dec_dim, cfg.patch_size ** 2 * cfg.num_classes)
    )

    total = 0
    for step in range(num_glimpses + 1):           # step=0 is the zero-step init
        n = step * cfg.patches_per_glimpse
        total += P.patch_embed(cfg.total_patches, cfg.patch_size, cfg.enc_dim)
        total += cfg.enc_depth * P.vit_block(n + 1, cfg.enc_dim, FFN_RATIO)
        total += P.linear(n + 1, cfg.enc_dim, cfg.dec_dim)   # decoder_embed
        total += dec_blocks_plus_head
    return total


def ame_gflops(num_glimpses: int, cfg: AMEConfig = AME) -> float:
    return round(ame_flops(num_glimpses, cfg) / 1e9, 2)
