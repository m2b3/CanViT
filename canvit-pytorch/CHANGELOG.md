# Changelog

Changes that reach users of the `canvit-pytorch` package, newest first. A release moves the
Unreleased entries under its version. Entries marked **Breaking** change what existing code or
saved files do; **Hub** entries are steps on the Hugging Face Hub that go with the release, in
order, each before or after it as the entry says, because released versions load each repo's
`main` revision.

## Unreleased

### Added

- `canvit_pytorch.rl`: learned viewing policies for the frozen CanViT-B (a viewpoint scorer that
  records the CanViT and probe it reads, the policy that follows it, and the training recipe of the
  published scorers, `python -m canvit_pytorch.rl.train`).
- `canvit_pytorch.evaluate` evaluates learned policies (ADE20K mIoU after every glimpse); the result
  records the scorer's repo, Hub commit and weights hash.
- `python -m canvit_pytorch.hub.publish policy` stages a policy-training run as a Hub repo.
- `canvit_pytorch.episode.take_glimpse`: CanViT's step on the glimpse at a viewpoint.
- `hub.repos.resolve`: where a Hub id of the organization loads from, a local mirror when
  `CANVIT_HUB_ROOT` names one.
- `hub.repos.Q_REGRESSION_POLICIES` and `POLICY_GRADIENT_POLICIES`, the published scorers;
  `hub.repos.released_segmenter`, the flagship and its released ADE20K probe for a scene size and
  canvas grid; `hub.loading.checkpoint_record`, the Hub commit and weights hash that identify a
  checkpoint.
- The TPU fine-tuning config takes the pretraining dataset of the checkpoint it starts from.
- Helpers that several modules had each written for themselves: `hub.repos.hub_id` (the Hub id of a
  repo given by its id, its name or its local-mirror directory), `preprocess.imagenet_denormalize_uint8`,
  `teacher.DINOv3Teacher.patch_grid` (patch features of images resized to an input size) and
  `benchmarks.ade20k.validation_set`.

### Hub

- Before the first release that contains `canvit_pytorch.rl`, which reads only its own scorer
  format, republish the learned policies (`canvit/qpolicy-ade20k-*`, `canvit/pgpolicy-ade20k-*`),
  which are in the format of the archived code that trained them. With that release's
  `canvit_pytorch.rl`, run
  `python -m canvit_pytorch.hub.republish_policies --release VERSION --out-dir STAGING`, review the
  staged repos, then run the same command with `--push`, which tags the commit each repo was staged
  from `original-format` for the archived code before uploading. No released version reads these
  repos, so no user breaks. Then delete `canvit_pytorch.hub.republish_policies`.

### Changed

- **Breaking**: evaluation results store paths and version strings in their metadata as `str`, so a
  result loads with `torch.load(..., weights_only=True)` on any Python. Code that called `Path`
  methods on result metadata wraps the value in `Path`.
- Attention heads merge with an explicit contiguous copy; eager outputs are bitwise unchanged, and
  the model now compiles under TorchTPU.
- A checkpoint whose weights do not match its model fails with an error naming the model class and
  the mismatched weights, without the 0.1 upgrade hint: a 0.1 checkpoint fails earlier, on its
  config.json, with that hint.
- `CanViTForImageClassification.from_pretrained_with_probe` fuses the probe with
  `canvit_core.readout.fuse_probe`, as the native packages do; the fused weights of every released
  checkpoint with the released DINOv3 probe are bitwise unchanged.
- `SegmentationProbe` computes in float32 also under autocast, as every evaluation already ran it,
  so its logits are float32 whatever the caller's precision. Under autocast it used to compute in
  bfloat16 unless the caller disabled autocast; float32 callers get bitwise the same logits.
- The TPU fine-tuning's environment, sharding and compilation logging moved from
  `specialize.in1k_tpu.train` to `specialize.in1k_tpu.diagnostics`.

### Removed

- **Breaking**: the 0.1 to 0.2 checkpoint converter, `python -m canvit_pytorch.legacy` and
  `canvit_pytorch.legacy.convert_checkpoint` (with `convert_state_dict` and
  `convert_canvit_config`). Every released checkpoint is in the 0.2 format on the Hub's `main`, and
  its 0.1 files remain at the revision `canvit-pytorch-0.1`. Loading a 0.1 checkpoint, or importing
  a name 0.2 renamed, still fails with an error that says what to do.
- `canvit_pytorch.hub.republish` and `canvit_pytorch.hub.reference`: the republication of the 0.1
  checkpoints in the 0.2 format, which is done.
- The `classifier` command of `python -m canvit_pytorch.hub.publish`, which published classifiers
  exported in the 0.1 format by a trainer outside this package (the two it published are on the
  Hub), and `canvit_pytorch.hub.cards.classifier_card`.
- **Breaking**: names that repeated what `canvit-core` owns; import them from `canvit-core`:
  `canvit_pytorch.model.config` (`CanViTConfig` stays importable from `canvit_pytorch`), the
  re-exports `BACKBONES`, `BackboneName` and `ViTSpec` of `canvit_pytorch.model.backbone`
  (`canvit_core.backbone`), `canvit_pytorch.model.classification.fuse_probe`
  (`canvit_core.readout.fuse_probe`, on NumPy arrays) and `canvit_pytorch.hub.loading.hub_file`
  (`canvit_core.checkpoint.hub_file`). `canvit_pytorch.model.backbone.vit` is now
  `canvit_pytorch.model.backbone`.
- **Breaking**: copies of the released geometry and of the helpers above; read them from their
  owners: `evaluate.config.SCENE_SIZE_PX`, `GLIMPSE_SIZE_PX` and `CANVAS_GRID_SIZE`,
  `specialize.in1k_tpu.data.SCENE_SIZE_PX` and `specialize.in1k_tpu.train.CANVAS_GRID_SIZE`
  (`hub.repos.RELEASED_*`); `evaluate.tasks.ade20k_segmentation.dinov3_patch_grid`, `probe_logits`
  and `ade20k_validation_set` (`DINOv3Teacher.patch_grid`, `SegmentationProbe` itself,
  `benchmarks.ade20k.validation_set`); `viz.live.publish.hub_name` (`hub.repos.hub_id`).
