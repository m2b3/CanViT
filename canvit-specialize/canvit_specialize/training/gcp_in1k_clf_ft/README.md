# CanViT IN1k finetuning (GCP TPU v6e)

Trains the IN1k checkpoint at [canvit/canvitb16-add-vpe-finetune-g128px-s512px-in1k-2026-04-06](https://huggingface.co/canvit/canvitb16-add-vpe-finetune-g128px-s512px-in1k-2026-04-06).
Requires GCP + SkyPilot + a TPU v6e-4 quota.

## Setup (laptop)

1. `uv tool install 'skypilot-nightly[gcp]'`, then `sky check gcp` green.
2. `gcloud auth application-default login`.
3. `~/.sky/config.yaml`:
   ```yaml
   gcp:
     remote_identity: SERVICE_ACCOUNT
   ```
4. GCS buckets named `${GCS_BUCKET_PREFIX}-${REGION}` for ImageNet TFRecords and `${GCS_BUCKET_PREFIX}-us-central1` for checkpoints. Edit `file_mounts.source` in the YAML to match.

## Launch

```bash
export COMET_API_KEY=$(cat ~/.config/comet_api_key.txt)
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
sky jobs launch canvit_specialize/training/gcp_in1k_clf_ft/sky-train-imagenet.yaml -y \
  --secret COMET_API_KEY --secret HF_TOKEN \
  --env GCS_BUCKET_PREFIX=your-prefix
```

Hyperparameters live in `sky-train-imagenet.yaml` `envs:`. Override via `--env LR=...`.

## Publish

```bash
uv run python scripts/push_finetuned.py --help
```
