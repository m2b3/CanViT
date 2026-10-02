from dataclasses import dataclass

from canvit_paper_exporter.flops import primitives as P
from canvit_paper_exporter.flops.primitives import FFN_RATIO


@dataclass(frozen=True)
class AdaGlimpseConfig:
    """AdaGlimpse (Pardyl et al. 2024, arXiv:2404.03482). ViT-B/16 encoder + 8-block
    decoder + CNN upsampling segmentation head.

    Encoder/decoder dims from the paper (§4). The CNN head is the
    `VisionTransformerUpHead` defined in `architectures/mae_utils.py:278-308`
    of `github.com/apardyl/AdaGlimpse @ 9b3b72d`: six Conv2d ops separated by
    four bilinear `nn.Upsample` stages (Upsample produces no matmul FLOPs).
    Each `cnn_layers` row is (H, W, c_in, c_out, kernel) of one Conv2d call,
    in code order: input projection at 14×14, four post-upsample 3×3 convs
    spanning 28→224, then a final 1×1 projecting to `num_classes`.
    """
    name: str = "AdaGlimpse"
    enc_dim: int = 768
    enc_depth: int = 12
    dec_dim: int = 512
    dec_depth: int = 8
    image_size: int = 224
    patch_size: int = 16
    glimpse_grid: int = 3
    num_classes: int = 150
    cnn_layers: tuple[tuple[int, int, int, int, int], ...] = (
        (14,  14,  512, 512, 3),
        (28,  28,  512, 256, 3),
        (56,  56,  256, 256, 3),
        (112, 112, 256, 256, 3),
        (224, 224, 256, 256, 3),
        (224, 224, 256, 150, 1),
    )

    @property
    def total_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def patches_per_glimpse(self) -> int:
        return self.glimpse_grid ** 2


ADAGLIMPSE = AdaGlimpseConfig()


def adaglimpse_flops(num_glimpses: int, cfg: AdaGlimpseConfig = ADAGLIMPSE) -> int:
    """Total AdaGlimpse FLOPs across `num_glimpses` glimpses.

    Single forward loop, no zero-step init. Decoder transformer operates on
    `(visible_tokens + 1 CLS + total_patches mask_tokens)` — NOT a constant
    `total_patches + 1`, because forward_decoder concatenates encoder output
    with ALL mask positions. CNN upsample head cost is constant per call.
    """
    dec_cnn_per_call = sum(P.conv2d(*layer) for layer in cfg.cnn_layers)
    total_enc = 0
    total_dec = 0
    for step in range(1, num_glimpses + 1):
        n = step * cfg.patches_per_glimpse
        total_enc += P.patch_embed(n, cfg.patch_size, cfg.enc_dim)
        total_enc += cfg.enc_depth * P.vit_block(n + 1, cfg.enc_dim, FFN_RATIO)
        dec_seq = n + 1 + cfg.total_patches
        total_dec += cfg.dec_depth * P.vit_block(dec_seq, cfg.dec_dim, FFN_RATIO)
        total_dec += dec_cnn_per_call
    return total_enc + total_dec


def adaglimpse_gflops(num_glimpses: int, cfg: AdaGlimpseConfig = ADAGLIMPSE) -> float:
    return round(adaglimpse_flops(num_glimpses, cfg) / 1e9, 2)
