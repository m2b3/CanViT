# Task specialization (`canvit_pytorch.specialize`)

Training [CanViT](../../README.md), the Canvas Vision Transformer, for downstream tasks (paper, Appendix D):
linear ADE20K segmentation probes on frozen features, and ImageNet-1k fine-tuning on Cloud TPU.

## Install

```bash
uv add "canvit-pytorch[specialize]"
```

## Using a trained probe

```python
from canvit_pytorch import SegmentationProbe
probe = SegmentationProbe.from_pretrained("canvit/probe-ade20k-40k-s512-c64-in21k")
logits = probe(features)  # [B, H, W, D] → [B, num_classes, H, W]
```

`canvit_pytorch.CanViTForSemanticSegmentation` pairs a CanViT with a probe on its canvas.

## ADE20K probes

Set `ADE20K_ROOT` to the ADEChallengeData2016 directory, and `COMET_API_KEY` and `COMET_WORKSPACE` for
experiment tracking (`cp .envrc.example .envrc`, then edit it).

```bash
# On the canvas of the flagship CanViT-B, with a 64×64 canvas grid
uv run python -m canvit_pytorch.specialize.ade20k canvas-probe --output-dir runs --canvas-grid-size 64
# On the patch features of DINOv3 ViT-S/16 at 128 px
uv run python -m canvit_pytorch.specialize.ade20k dinov3-probe --output-dir runs --variant vits16 --input-size-px 128
```

`--help` lists every option; the defaults are the paper's protocol. The SLURM job scripts
`slurm/specialize/train_ade20k_canvas_probe.sbatch` and `train_ade20k_dinov3_probe.sbatch` write under
`$CHECKPOINTS_DIR/canvit-ade20k-probes`.

Each run writes a new directory holding:

- `probe_step<N>/`: the probe with the highest validation mIoU on the last feature map (for a canvas probe,
  the canvas after the last glimpse), which `SegmentationProbe.from_pretrained` loads;
- `record.json`: its step, its validation mIoU on each feature map, the run's configuration and provenance.
  `canvit_pytorch.specialize.ade20k.record.ProbeRecord` defines the fields.

## ImageNet-1k fine-tuning on Cloud TPU

See [`in1k_tpu/README.md`](../canvit_pytorch/specialize/in1k_tpu/README.md).
