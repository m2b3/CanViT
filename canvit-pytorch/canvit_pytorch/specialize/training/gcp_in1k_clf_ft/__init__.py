"""CanViT IN1k classification finetuning on GCP TPU v6e.

SPMD training on a TPU v6e-4 slice. Launch via the co-located SkyPilot
config `sky-train-imagenet.yaml`; requires the tpu/ environment (`uv sync --project tpu`, done by setup_tpu.sh).
See `./README.md` for setup.
"""
