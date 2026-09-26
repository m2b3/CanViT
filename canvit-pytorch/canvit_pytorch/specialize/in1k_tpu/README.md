# ImageNet-1k fine-tuning on Cloud TPU

Fine-tunes CanViT-B, the Canvas Vision Transformer, for ImageNet-1k classification on a TPU v6e-4 slice
(paper, Appendix D.4). This code trained
[canvit/canvitb16-add-vpe-finetune-g128px-s512px-in1k-2026-04-06](https://huggingface.co/canvit/canvitb16-add-vpe-finetune-g128px-s512px-in1k-2026-04-06).
It needs Google Cloud, SkyPilot and a TPU v6e-4 quota.

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

From `canvit-pytorch/`, which SkyPilot ships to the VM as the working directory:

```bash
export COMET_API_KEY=$(cat ~/.config/comet_api_key.txt)
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
sky jobs launch canvit_pytorch/specialize/in1k_tpu/sky-train-imagenet.yaml -y \
  --secret COMET_API_KEY --secret HF_TOKEN \
  --env GCS_BUCKET_PREFIX=your-prefix
```

The YAML's `envs:` hold the hyperparameters of the published checkpoint; override them with `--env LR=...`.
`train.py --help` describes each option.

The classifier loads the pretrained checkpoint from the Hub in the canvit-pytorch 0.2 format.
Checkpoints (`latest.pt`, `best.pt`, `init.pt`) hold `CanViTForImageClassification.state_dict()` and the
optimizer state.
