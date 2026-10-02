# Pretraining (`canvit_pytorch.pretrain`)

CanViT is pretrained by passive-to-active dense latent distillation. A frozen
DINOv3 ViT-B/16 produces patch and CLS features for each 512 px training
scene. CanViT predicts the scene-wide patch features from its canvas and the
teacher CLS feature from its recurrent CLS token after every glimpse. The
defaults of `PretrainingConfig` are the released CanViT-B geometry and
schedule.

The workflow has three stages: build an immutable image index, export teacher
features into shards, then train from those shards. All commands below run
from `canvit-pytorch/`.

## Environment

Install the pretraining dependencies and configure the paths used by the
commands and SLURM wrappers:

```bash
uv sync --extra pretrain
cp .envrc.example .envrc
```

Edit `.envrc` before loading it. The relevant variables are:

- `IN21K_IMAGE_DIR`: class-per-directory training images used to build the
  index and to load scenes while training.
- `IMAGENET_VAL`: ImageNet-1k validation images in class-per-directory form;
  pretraining uses them for validation metrics.
- `PROJECT_STORAGE`, `CHECKPOINTS_DIR`, `FEATURES_DIR` and `INDEX_DIR`:
  storage locations for checkpoints, teacher shards and the index. The example
  file derives the latter three from `PROJECT_STORAGE`.
- `HF_TOKEN`: access to the gated DINOv3 teacher.
- `COMET_API_KEY` and `COMET_WORKSPACE`: experiment tracking credentials.

Load the file with `direnv allow` or `source .envrc`. SLURM wrappers source
`.envrc` themselves through `slurm/env.sh`.

## 1. Index the images

`features.index` scans a class-per-directory tree, writes `path`,
`class_name` and `class_idx` columns, and shuffles the rows by default. The
index is immutable: shards record its hash, so choose a new output path for a
new dataset or ordering.

```bash
uv run python -m canvit_pytorch.pretrain.features.index \
  --image-root "$IN21K_IMAGE_DIR" \
  --out "$INDEX_DIR/in21k-shuffled.parquet"
```

The index stores its row count in Parquet metadata. Use that count when
checking the completed shard tree:

```bash
expected_images="$(uv run python -c 'import pyarrow.parquet as pq, sys; print(pq.read_metadata(sys.argv[1]).num_rows)' \
  "$INDEX_DIR/in21k-shuffled.parquet")"
```

## 2. Export teacher features

The exporter writes `out_dir/shards/<id>.pt`. A shard contains DINOv3 ViT-B/16
features of 512 px scenes, image paths, class indices, decoded-pixel hashes,
and provenance. It writes a shard atomically and skips shards that already
exist, so resubmitting the export resumes completed work.

The supplied array wrapper divides the shard range into tasks and uses the
paths derived from `.envrc`:

```bash
sbatch slurm/pretrain/export_features.sh
```

For another class-per-directory tree, set `DATASET` and `IMAGE_ROOT` before
submitting. The wrapper writes to
`$FEATURES_DIR/$DATASET/dinov3_vitb16/512/` and accepts `INDEX` and `OUT_DIR`
overrides. The direct command has the same options and is available through
`--help`:

```bash
uv run python -m canvit_pytorch.pretrain.features.export --help
```

Check the complete tree before training. The check requires contiguous shard
IDs, the expected row count, identical index/teacher/geometry metadata, and
zero failed image decodes:

```bash
uv run python -m canvit_pytorch.pretrain.features.check \
  --shards-dir "$FEATURES_DIR/in21k/dinov3_vitb16/512/shards" \
  --expected-images "$expected_images"
```

Failed decodes are recorded as failed indices and their feature tensors are
NaN. The check is the gate that prevents those rows from entering training.

## 3. Train

The SLURM wrapper supplies the required dataset and storage arguments. Each
array task trains `steps_per_job` optimizer steps, saves a checkpoint, and
exits; the next task resumes the same run. The default array covers the
released two-million-step schedule.

```bash
sbatch slurm/pretrain/train.sbatch
```

Set `DATASET=in1k` and `IMAGES_DIR=/path/to/in1k/train` for an ImageNet-1k
training tree. Set `RUN_NAME` to resume a named run. Extra arguments are
passed to `python -m canvit_pytorch.pretrain`; inspect the complete
configuration with:

```bash
uv run python -m canvit_pytorch.pretrain --help
```

The ablation wrapper accepts a slug from that help output, computes the
shorter ablation schedule, and submits the corresponding array:

```bash
bash slurm/pretrain/ablation.sh no-reads
```

## Checkpoints

`$CHECKPOINTS_DIR/<run-name>/` contains `step-<N>.pt` files and a
`latest.pt` symlink to the newest file. A training checkpoint stores the
`canvit-training-checkpoint-d0354a31-2d71-45e9-b2ad-ded890170034` format,
the `CanViTForPretraining` state, its `CanViTConfig`, teacher geometry,
dataset, optimizer and scheduler state, step, tracking key, and the per-job
configuration and provenance records. It is a resume checkpoint; publishing a
model repository is a separate operation.

If a job fails, the run directory contains `FAILED`. Fix the underlying cause,
then remove that marker before resubmitting the run. A job that sees the marker
stops and cancels the remaining array tasks.
