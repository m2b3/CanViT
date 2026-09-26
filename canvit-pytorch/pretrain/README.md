# CanViT-pretrain

Passive-to-active dense latent distillation of [CanViT](https://github.com/m2b3/CanViT-PyTorch) ([arXiv:2603.22570](https://arxiv.org/abs/2603.22570)) from [DINOv3](https://github.com/facebookresearch/dinov3) ([arXiv:2508.10104](https://arxiv.org/abs/2508.10104)).

Originally designed to run on [the Nibi SLURM cluster](https://docs.alliancecan.ca/wiki/Nibi) using its [hosted ImageNet-21k `winter21_whole` replica](https://docs.alliancecan.ca/wiki/ImageNet).

## Setup

```bash
cp .envrc.example .envrc && direnv allow
# Edit .envrc to adapt to your environment.
```

Please ensure that `HF_TOKEN`, `COMET_API_KEY`, and `COMET_WORKSPACE` are set.

## Run

Export DINOv3 teacher features once:

```bash
uv run python scripts/build_shuffled_index.py \
  --image-root $IN21K_IMAGE_DIR --index-dir $INDEX_DIR --dataset in21k
sbatch --array=0-99%20 slurm/export_features.sh
```

Pretraining:

```bash
sbatch slurm/train.sbatch [--flag value ...]
```

Ablations:

```bash
bash slurm/ablations/baseline.sh
bash slurm/ablations/no-bptt.sh
# ...
```

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
