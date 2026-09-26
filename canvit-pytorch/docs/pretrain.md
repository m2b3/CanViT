# Pretraining (`canvit_pytorch.pretrain`)

CanViT (the Canvas Vision Transformer) is pretrained by policy-agnostic
passive-to-active dense latent distillation (paper, Section 5): after every
glimpse, it predicts the frozen DINOv3 ViT-B/16 teacher's features of the whole
512 px scene, patches from the canvas and the CLS token from its recurrent CLS
token. The defaults of `PretrainingConfig` are CanViT-B's hyperparameters.

The paper's runs used [the Nibi SLURM cluster](https://docs.alliancecan.ca/wiki/Nibi)
and its [hosted ImageNet-21k `winter21_whole` replica](https://docs.alliancecan.ca/wiki/ImageNet).

## Setup

```bash
cp .envrc.example .envrc && direnv allow   # then set the paths for your machine
```

`HF_TOKEN`, `COMET_API_KEY` and `COMET_WORKSPACE` must be set.

## 1. Precomputed teacher features

```bash
uv run python -m canvit_pytorch.pretrain.features.index \
  --image-root $IN21K_IMAGE_DIR --out $INDEX_DIR/in21k-shuffled.parquet
sbatch slurm/pretrain/export_features.sh
uv run python -m canvit_pytorch.pretrain.features.check \
  --shards-dir $FEATURES_DIR/in21k/dinov3_vitb16/512/shards --expected-images <images in the index>
```

## 2. Pretraining

```bash
sbatch slurm/pretrain/train.sbatch                 # CanViT-B, 2M steps as a chain of jobs
bash slurm/pretrain/ablation.sh no-reads           # one of the paper's ablations
uv run python -m canvit_pytorch.pretrain --help    # every setting
```

Each job trains `--steps-per-job` steps and saves a checkpoint; the next job
resumes it. Checkpoints of a run live in `$CHECKPOINTS_DIR/<run name>/`, with
`latest.pt` pointing to the newest.
