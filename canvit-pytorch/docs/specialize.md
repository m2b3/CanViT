# Task specialization (`canvit_pytorch.specialize`)

This package trains linear ADE20K segmentation probes on frozen feature maps
and fine-tunes CanViT end to end for ImageNet-1k classification on a TPU
slice. The probe commands run from `canvit-pytorch/`; the TPU workflow is
documented in [`in1k_tpu/README.md`](../canvit_pytorch/specialize/in1k_tpu/README.md).
Released CanViT checkpoints and probes are listed in the [Hugging Face
organization](https://huggingface.co/canvit).

## Install and configure

From `canvit-pytorch/`, install the specialization dependencies:

```bash
uv sync --extra specialize
```

Set `ADE20K_ROOT` to the `ADEChallengeData2016` directory. It must contain
`images/training`, `images/validation`, `annotations/training` and
`annotations/validation`. Probe training also needs the experiment-tracking
variables `COMET_API_KEY` and `COMET_WORKSPACE`. Copy `.envrc.example`, set
these values and load it before launching a probe:

```bash
cp .envrc.example .envrc
source .envrc
```

The evaluation transform resizes each validation image and label map to the
configured square scene size without cropping. Training uses the configured
scale jitter, square crop, horizontal flip and photometric augmentation.

## ADE20K probes

`canvas-probe` decodes the canvas patches after every glimpse of a frozen
CanViT episode. Its probe includes a LayerNorm because CanViT's
canvas features are not already normalized. `dinov3-probe` decodes one map of
DINOv3 patch features; DINOv3's final outputs are already layer-normalized, so
that probe skips the extra LayerNorm. Each probe is a 1×1 classifier trained
with the segmentation evaluation protocol.

The released CanViT-B probe at a 64×64 canvas can be loaded directly:

```python
from canvit_pytorch import SegmentationProbe

probe = SegmentationProbe.from_pretrained("canvit/probe-ade20k-40k-s512-c64-in21k")
logits = probe(features)  # [B, H, W, D] -> [B, 150, H, W]
```

`CanViTForSemanticSegmentation` composes a pretrained CanViT with this probe;
its `logits` method drops the canvas registers, applies the probe's readout
normalization and returns `[B, 150, G, G]` logits.

Train a canvas probe on the released CanViT-B at a 64×64 canvas:

```bash
uv run python -m canvit_pytorch.specialize.ade20k canvas-probe \
  --output-dir runs \
  --canvas-grid-size 64
```

Train the passive DINOv3 ViT-S/16 baseline at a 128 px input resolution:

```bash
uv run python -m canvit_pytorch.specialize.ade20k dinov3-probe \
  --output-dir runs \
  --variant vits16 \
  --input-size-px 128
```

The command defaults are the paper's probe protocol. `--help` lists all
options, including the pretrained checkpoint, scene and glimpse geometry,
viewing policies, optimizer schedule, feature dtype and validation interval.
The canvas probe's `--pretrained-repo`, `--canvas-grid-size` and
`--glimpse-size-px` must describe the checkpoint and probe geometry you intend
to evaluate.

For a scheduled run, pass the same options after the corresponding wrapper:

```bash
sbatch slurm/specialize/train_ade20k_canvas_probe.sbatch --canvas-grid-size 64
sbatch slurm/specialize/train_ade20k_dinov3_probe.sbatch \
  --variant vits16 --input-size-px 128
```

Each run creates a new subdirectory under `--output-dir`. The directory holds:

- `probe_step<N>/`: the selected probe, saved in the format loaded by
  `SegmentationProbe.from_pretrained`;
- `record.json`: the selected optimizer step, validation mIoU for every feature
  map, complete configuration, tracking key and provenance.

The selected probe is the checkpoint with the highest validation mIoU on the
last feature map. For a canvas probe, that map is the canvas after the final
configured glimpse. Older selected probe directories are removed when a later
validation becomes the new selection.

## ImageNet-1k fine-tuning on TPU

See [`in1k_tpu/README.md`](../canvit_pytorch/specialize/in1k_tpu/README.md) for
the separate TPU environment, TFRecord layout, launch configuration and
checkpoint format.
