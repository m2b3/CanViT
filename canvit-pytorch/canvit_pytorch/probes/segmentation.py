"""Segmentation probe: LN -> Dropout -> BN -> Conv1x1.

Architecture follows DINOv3. Loadable through ``PyTorchModelHubMixin`` from a
repo ID or local checkpoint directory.

Example::

    probe = SegmentationProbe.from_pretrained("<org>/probe-ade20k-...")
    logits = probe(features)  # [B, H, W, D] -> [B, num_classes, H, W]
"""

from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn
from torch.nn import functional as F


class SegmentationProbe(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="canvit-pytorch",
    repo_url="https://github.com/m2b3/CanViT-PyTorch",
):
    """Linear segmentation head on spatial features.

    Input: [B, H, W, D] spatial features (canvas tokens or DINOv3 patches).
    Output: [B, num_classes, H, W] logits at the input spatial resolution.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_ln: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.dropout_p = dropout
        self.use_ln = use_ln
        self.ln: nn.Module = nn.LayerNorm(embed_dim) if use_ln else nn.Identity()
        self.bn = nn.BatchNorm2d(embed_dim)
        self.dropout = nn.Dropout2d(dropout)
        self.conv = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        nn.init.normal_(self.conv.weight, mean=0, std=0.01)
        assert self.conv.bias is not None
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """[B, H, W, D] -> [B, num_classes, H, W]."""
        B, H, W, D = x.shape
        assert D == self.embed_dim, f"Expected embed_dim={self.embed_dim}, got {D}"
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.dropout(x)
        x = self.bn(x)
        return self.conv(x)

    def predict(self, x: Tensor, target_size: tuple[int, int]) -> Tensor:
        """Forward + bilinear upsample to target resolution."""
        return F.interpolate(self(x), size=target_size, mode="bilinear", align_corners=False)
