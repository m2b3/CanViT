"""AdaGlimpse's hyperparameters, and the configuration the paper's architecture control trained."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AdaGlimpseConfig:
    """AdaGlimpse's masked-autoencoder ViT on square scenes, with readouts into a teacher's feature space.

    The defaults are AdaGlimpse's ViT-B (`mae_vit_base_patch16_dec512d8b` upstream).
    The encoder places each patch by the scene box it covers, so the scene size
    fixes only the decoder's output grid: a position per patch-sized cell.
    AdaGlimpse cuts a glimpse into glimpse_tokens_per_side² square crops and
    resizes each to a patch, so a glimpse costs that many tokens whatever area it covers.
    """

    scene_size_px: int
    glimpse_tokens_per_side: int
    teacher_dim: int
    patch_size_px: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    decoder_embed_dim: int = 512
    decoder_depth: int = 8
    decoder_num_heads: int = 16
    mlp_ratio: int = 4

    def __post_init__(self) -> None:
        assert self.scene_size_px % self.patch_size_px == 0, (self.scene_size_px, self.patch_size_px)
        assert self.glimpse_tokens_per_side > 0 and self.teacher_dim > 0, self
        assert self.embed_dim % self.num_heads == 0, (self.embed_dim, self.num_heads)
        assert self.decoder_embed_dim % self.decoder_num_heads == 0, (self.decoder_embed_dim, self.decoder_num_heads)
        # The patch-box embedding gives each of a box's four coordinates a sine and a cosine half of a quarter.
        assert self.embed_dim % 8 == 0, self.embed_dim

    @property
    def output_grid_size(self) -> int:
        """Side of the decoder's output grid."""
        return self.scene_size_px // self.patch_size_px

    @property
    def tokens_per_glimpse(self) -> int:
        return self.glimpse_tokens_per_side**2


ARCHITECTURE_CONTROL = AdaGlimpseConfig(scene_size_px=512, glimpse_tokens_per_side=2, teacher_dim=768)
"""The baseline of the paper's architecture control (`supp:architecture-control`): AdaGlimpse's published
tokenization of a glimpse, on the scenes CanViT-B sees, with readouts into DINOv3 ViT-B/16's feature space."""
