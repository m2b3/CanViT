"""Validate exported features against fresh inference.

Usage (in interactive session):
    source slurm/env.sh
    uv run python scripts/validate_features.py \
        --shard $FEATURES_DIR/dinov3_vitb16/512/shards/00000.pt \
        --image-root $IN21K_IMAGE_DIR \
        --teacher-repo-id facebook/dinov3-vitb16-pretrain-lvd1689m \
        --idx 42
"""

import argparse
import sys
from pathlib import Path

import torch

# Import exact same dataset class as export
sys.path.insert(0, str(Path(__file__).parent))
from canvit_pytorch.teacher import load_teacher
from export_in21k_features import ImageDataset


def compare(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    diff = (a.float() - b.float()).abs()
    print(f"  {name}: max={diff.max().item():.6f}, mean={diff.mean().item():.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--teacher-repo-id", type=str, required=True)
    parser.add_argument("--idx", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load shard
    shard = torch.load(args.shard, map_location="cpu", weights_only=False, mmap=True)
    stored_patches = shard["patches"][args.idx]
    rel_path = shard["paths"][args.idx]
    print(f"Image: {rel_path}")
    print(f"Stored dtype: {stored_patches.dtype}")

    # Load image using EXACT same Dataset class as export
    dataset = ImageDataset(args.image_root, [rel_path], shard["image_size"])
    img_tensor, idx, success, img_hash = dataset[0]
    assert success, f"Failed to load image: {rel_path}"
    print(f"Image hash: {img_hash}")
    print(f"Stored hash: {shard['image_hashes'][args.idx]}")
    assert img_hash == shard["image_hashes"][args.idx], "Hash mismatch!"
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Load teacher (same as export: HuggingFace Hub)
    teacher = load_teacher(args.teacher_repo_id, device)

    with torch.no_grad():
        # Run inference at different precisions
        print("\n=== Fresh inference ===")

        # float32 inference (ground truth)
        feats_f32 = teacher.forward_norm_features(img_tensor)
        patches_f32 = feats_f32.patches[0].cpu()  # float32
        print(f"float32 output range: [{patches_f32.min():.4f}, {patches_f32.max():.4f}]")

        # bfloat16 autocast → float16 storage (what export uses)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            feats_bf16 = teacher.forward_norm_features(img_tensor)
            patches_autocast_raw = feats_bf16.patches[0].cpu()  # f32 from autocast
            patches_export_path = patches_autocast_raw.to(torch.float16)  # matches export: bf16 autocast → f16 storage

        print("\n=== Precision comparison ===")

        # Cast ground truth to different dtypes
        patches_to_bf16 = patches_f32.to(torch.bfloat16)
        patches_to_f16 = patches_f32.to(torch.float16)

        print("Quantization error (f32 → dtype → f32):")
        compare("f32 → bf16", patches_f32, patches_to_bf16.float())
        compare("f32 → f16 ", patches_f32, patches_to_f16.float())

        print("\nAutocast vs f32 (same run):")
        compare("bf16 autocast raw vs f32  ", patches_f32, patches_autocast_raw)
        compare("bf16 autocast→f16 vs f32  ", patches_f32, patches_export_path)

        print("\nStored vs fresh:")
        compare("stored vs f32             ", stored_patches, patches_f32)
        compare("stored vs f32→f16         ", stored_patches, patches_to_f16)
        compare("stored vs autocast raw    ", stored_patches, patches_autocast_raw)
        compare("stored vs autocast→f16   ", stored_patches, patches_export_path)  # exact export path

        # Run autocast twice to check determinism
        print("\n=== Determinism check ===")
        with torch.autocast(device.type, dtype=torch.bfloat16):
            run1 = teacher.forward_norm_features(img_tensor).patches[0].cpu()
            run2 = teacher.forward_norm_features(img_tensor).patches[0].cpu()
        compare("run1 vs run2 (same session)", run1, run2)

        # dtype precision info
        print("\n=== Dtype info ===")
        print(f"bfloat16: 8 exp bits, 7 mantissa bits, eps={torch.finfo(torch.bfloat16).eps:.6f}")
        print(f"float16:  5 exp bits, 10 mantissa bits, eps={torch.finfo(torch.float16).eps:.6f}")
        print(f"float32: 8 exp bits, 23 mantissa bits, eps={torch.finfo(torch.float32).eps:.8f}")


if __name__ == "__main__":
    main()
