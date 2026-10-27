"""Analytic forward FLOPs of AdaGlimpse after each glimpse.

One multiply-accumulate counts as two FLOPs. The counts cover what
torch.utils.flop_counter counts: matrix products (linear layers, both products
of attention) and the patch embedding. Normalization, softmax, activations,
positional embeddings and residual additions are left out. AdaGlimpse encodes
every patch it has seen after each glimpse, so the cost of a glimpse grows with
the number of glimpses before it.
"""

from adaglimpse_nnx.config import AdaGlimpseConfig


def linear_flops(*, num_tokens: int, in_dim: int, out_dim: int) -> int:
    return 2 * num_tokens * in_dim * out_dim


def encoder_flops(config: AdaGlimpseConfig, *, num_glimpses: int) -> int:
    """Encoding the patches of num_glimpses glimpses: their embedding, then the blocks over them and the class token."""
    num_patches = _num_patches(config, num_glimpses=num_glimpses)
    patch_embedding = linear_flops(num_tokens=num_patches, in_dim=3 * config.patch_size_px**2, out_dim=config.embed_dim)
    block = _block_flops(num_tokens=1 + num_patches, dim=config.embed_dim, mlp_ratio=config.mlp_ratio)
    return patch_embedding + config.depth * block


def decoder_flops(config: AdaGlimpseConfig, *, num_glimpses: int) -> int:
    """Decoding that encoding into the output grid's features: the decoder's input projection, then its blocks over
    the encoded tokens and a mask token per grid cell."""
    num_encoded = 1 + _num_patches(config, num_glimpses=num_glimpses)
    embed = linear_flops(num_tokens=num_encoded, in_dim=config.embed_dim, out_dim=config.decoder_embed_dim)
    block = _block_flops(
        num_tokens=num_encoded + config.output_grid_size**2, dim=config.decoder_embed_dim, mlp_ratio=config.mlp_ratio,
    )
    return embed + config.decoder_depth * block


def segmentation_flops(config: AdaGlimpseConfig, *, num_glimpses: int, num_classes: int) -> int:
    """A forward after num_glimpses glimpses, then a linear segmentation probe on each cell of the decoded grid.

    This is the per-glimpse cost behind the architecture control's ADE20K
    FLOPs; the teacher readouts are not run.
    """
    probe = linear_flops(
        num_tokens=config.output_grid_size**2, in_dim=config.decoder_embed_dim, out_dim=num_classes,
    )
    return encoder_flops(config, num_glimpses=num_glimpses) + decoder_flops(config, num_glimpses=num_glimpses) + probe


def _num_patches(config: AdaGlimpseConfig, *, num_glimpses: int) -> int:
    assert num_glimpses > 0, num_glimpses
    return num_glimpses * config.tokens_per_glimpse


def _block_flops(*, num_tokens: int, dim: int, mlp_ratio: int) -> int:
    return (
        linear_flops(num_tokens=num_tokens, in_dim=dim, out_dim=3 * dim)  # queries, keys, values
        + 4 * num_tokens * num_tokens * dim  # query-key scores, then the weighted sum of values
        + linear_flops(num_tokens=num_tokens, in_dim=dim, out_dim=dim)  # output
        + 2 * linear_flops(num_tokens=num_tokens, in_dim=dim, out_dim=mlp_ratio * dim)  # the MLP's two layers
    )
