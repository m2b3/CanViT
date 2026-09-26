"""ImageNet normalization, the input convention of CanViT and of its DINOv3 teacher."""

from torch import Tensor
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_normalize(image: Tensor) -> Tensor:
    """[..., 3, H, W] in [0, 1] -> normalized."""
    return (image - image.new_tensor(IMAGENET_MEAN).view(3, 1, 1)) / image.new_tensor(IMAGENET_STD).view(3, 1, 1)


def imagenet_denormalize(image: Tensor) -> Tensor:
    """Inverse of imagenet_normalize, clamped to [0, 1]."""
    return (image * image.new_tensor(IMAGENET_STD).view(3, 1, 1) + image.new_tensor(IMAGENET_MEAN).view(3, 1, 1)).clamp(0, 1)


def preprocess(size: int) -> transforms.Compose:
    """PIL image -> normalized [3, size, size] tensor: resize the short side, center-crop."""
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_labels(size: int) -> transforms.Compose:
    """PIL label map -> [1, size, size] integer tensor, with the geometry of preprocess(size) and no interpolation."""
    return transforms.Compose([
        transforms.Resize(size, transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(size),
        transforms.PILToTensor(),
    ])
