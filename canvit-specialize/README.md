# CanViT-specialize

Training loops for [CanViT](https://github.com/m2b3/CanViT-PyTorch) downstream probes (ADE20K segmentation) and IN1k finetuning.

## Install

```bash
uv add "canvit-specialize @ git+https://github.com/m2b3/CanViT-specialize.git"
```

For TPU finetuning, see [`gcp_in1k_clf_ft/README.md`](canvit_specialize/training/gcp_in1k_clf_ft/README.md).

## Using a pre-trained probe

```python
from canvit_pytorch import SegmentationProbe
probe = SegmentationProbe.from_pretrained("canvit/probe-ade20k-40k-s512-c64-in21k")
logits = probe(features)  # [B, H, W, D] → [B, num_classes, H, W]
```

For the fused **CanViT + probe** pair, see `canvit_pytorch.CanViTForSemanticSegmentation`.

## Training

`COMET_API_KEY`, `COMET_WORKSPACE`, and `ADE20K_ROOT` must be set before training.

```bash
cp .envrc.example .envrc && direnv allow
# Edit .envrc to point at your dataset / Comet workspace.
```

### ADE20K segmentation probe (frozen CanViT)

```bash
uv run python -m canvit_specialize.training.ade20k train \
  --scene-size 512 --canvas-grid 64
```

### DINOv3 baseline probe

```bash
uv run python -m canvit_specialize.training.ade20k train-dinov3-probe
```

### IN1k classification finetuning on GCP TPU v6e

See [`canvit_specialize/training/gcp_in1k_clf_ft/README.md`](canvit_specialize/training/gcp_in1k_clf_ft/README.md).

## Citation

```bibtex
@article{berreby2026canvit,
  title={CanViT: Toward Active-Vision Foundation Models},
  author={Berreby, Yoha{\"i}-Eliel and Du, Sabrina and Durand, Audrey and Krishna, B. Suresh},
  year={2026},
  eprint={2603.22570},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.22570}
}
```

## License

MIT. See [LICENSE](LICENSE) for details.
