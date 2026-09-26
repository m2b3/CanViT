"""CanViT IN1k classification finetuning on GCP TPU v6e.

SPMD training on a TPU v6e-4 slice. Launch via the co-located SkyPilot
config `sky-train-imagenet.yaml`; requires `uv sync --group gcp-in1k-finetune`.
See `./README.md` for setup.
"""
