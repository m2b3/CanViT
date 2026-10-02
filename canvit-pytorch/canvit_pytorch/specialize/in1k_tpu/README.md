# ImageNet-1k fine-tuning on TPU

This workflow fine-tunes CanViT-B end to end for ImageNet-1k classification on
a TPU v6e-4 slice with PyTorch/XLA SPMD. The default task configuration is the
published LP-FT run, whose checkpoint is
`canvit/canvitb16-add-vpe-finetune-g128px-s512px-in1k-2026-04-06`.

The task uses the pretraining checkpoint
`canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02` and
the released DINOv3 ViT-B/16 ImageNet-1k CLS probe. It samples four
glimpses per scene with the full-then-random policy, computes classification
loss after every glimpse, and uses four-glimpse truncated backpropagation
through time. The code loads
the pretrained model and probe from the Hub when the task starts.

The pretrained model and the released fine-tuning checkpoints and probes are
listed in the [Hugging Face organization](https://huggingface.co/canvit). The
pretrained Hub model uses the canvit-pytorch 0.2 checkpoint format; the
training files below are the fine-tuning trainer's separate resume format.

## Prerequisites

The launch machine needs:

- `uv`;
- the SkyPilot command with cloud support;
- `gcloud` application-default credentials;
- access to a TPU v6e-4 slice; and
- two object-storage buckets for the ImageNet TFRecords and checkpoints.

Install the launch command and check the accelerator account:

```bash
uv tool install 'skypilot-nightly[gcp]'
sky check gcp
gcloud auth application-default login
```

Configure the worker identity once in `~/.sky/config.yaml`:

```yaml
gcp:
  remote_identity: SERVICE_ACCOUNT
```

The task uses a region-specific data bucket named
`${GCS_BUCKET_PREFIX}-${REGION}` and a checkpoint bucket named
`${GCS_BUCKET_PREFIX}-us-central1`. Edit `file_mounts.source` in
`sky-train-imagenet.yaml` to the checkpoint bucket you own. The data bucket
must contain these paths under `datasets/imagenet/`:

```text
datasets/imagenet/train-*
datasets/imagenet/validation-*
```

The loader expects the `train-*` and `validation-*` files to be direct
children of the data directory. It decodes the TFRecord fields
`image/encoded` and `image/class/label`; labels are one-based in the records
and become zero-based class indices in the trainer.

## Launch

Run the command from the monorepo root. The YAML ships the root checkout so
the TPU environment can resolve the local `canvit-core` package and the
PyTorch build hook can reach the root `LICENSE` and `tools/license_build.py`.
The command below forwards `COMET_API_KEY` and `HF_TOKEN` from the launch
shell as SkyPilot secrets. Set those variables before using the command.
The published CanViT checkpoints are public; a Hub token supports authenticated
downloads.

```bash
sky jobs launch canvit-pytorch/canvit_pytorch/specialize/in1k_tpu/sky-train-imagenet.yaml -y \
  --secret COMET_API_KEY \
  --secret HF_TOKEN \
  --env GCS_BUCKET_PREFIX=your-prefix
```

The YAML's `setup` block enters `canvit-pytorch/` and runs
`canvit_pytorch/specialize/in1k_tpu/setup_tpu.sh`. The setup script installs
Python 3.12 and the separate `tpu/` environment, configures the shared library
path needed by `torch_xla`, installs `gcsfuse`, and installs the local
`canvit-pytorch` package without resolving the main project's CUDA-indexed
dependencies. The TPU project installs editable `../../canvit-core` before
that no-dependency package install.

The YAML passes its declared values to `train.py`, sets 32 data-loader workers
and logs every 100 optimizer steps. Run the trainer's help command from
`canvit-pytorch/` to inspect every option:

```bash
uv run --project tpu --no-sync python -m canvit_pytorch.specialize.in1k_tpu.train --help
```

Override declared environment values with additional `--env` arguments. The
optional values are:

- `RUN_NAME`: experiment name and fallback checkpoint-directory name;
- `INIT_FROM`: classifier checkpoint whose model weights initialize a new run;
- `EVAL_C2F`: any nonempty value to add coarse-to-fine validation;
- `EVAL_N_GLIMPSES`: validation episode length; and
- `EARLY_STOP_DELTA`: stop when validation accuracy falls below the best by
  this amount.

When `INIT_FROM` is set, the checkpoint's model weights load at step 0 and a
new optimizer starts. Otherwise the task resumes model and optimizer state
from `/ckpt/checkpoints/<task-id>/latest.pt`; set `RUN_NAME` when a stable
checkpoint directory is needed across manual executions.

## Checkpoints

The trainer writes `latest.pt`, `best.pt` and `init.pt` under its checkpoint
directory. Each file is a `torch.save` dictionary with these fields:

```text
step
best_val_acc
comet_key
model_state_dict
optimizer_state_dict
```

`latest.pt` is updated at the configured checkpoint interval and after
validation. `best.pt` is written when the primary validation accuracy after
the final glimpse improves. `init.pt` records the initialized classifier after
the first compiled step. The checkpoint loader restores the classifier and
optimizer state from `latest.pt`; `INIT_FROM` uses only `model_state_dict` and
starts a new optimizer.
