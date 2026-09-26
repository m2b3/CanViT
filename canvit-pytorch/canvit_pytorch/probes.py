"""Linear probes that decode frozen spatial features, CanViT's canvas or a passive ViT's patch grid."""

from torch import Tensor, nn

from canvit_pytorch.hub.loading import HubMixin


class SegmentationProbe(nn.Module, HubMixin):
    """[B, H, W, D] features -> [B, num_classes, H, W] logits: LayerNorm, Dropout, BatchNorm, 1×1 convolution.

    The head follows DINOv3's linear segmentation evaluation. Features that are
    already layer-normalized (DINOv3's outputs) skip the LayerNorm (use_ln=False).
    """

    def __init__(self, *, embed_dim: int, num_classes: int, dropout: float, use_ln: bool) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_ln = use_ln
        self.ln: nn.Module = nn.LayerNorm(embed_dim) if use_ln else nn.Identity()
        self.bn = nn.BatchNorm2d(embed_dim)
        self.dropout = nn.Dropout2d(dropout)
        self.conv = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        nn.init.normal_(self.conv.weight, mean=0, std=0.01)
        assert self.conv.bias is not None
        nn.init.zeros_(self.conv.bias)

    def forward(self, features: Tensor) -> Tensor:
        B, H, W, D = features.shape
        assert D == self.embed_dim, f"probe expects {self.embed_dim}-dim features, got {D}"
        x = self.ln(features).permute(0, 3, 1, 2).contiguous()
        return self.conv(self.bn(self.dropout(x)))
