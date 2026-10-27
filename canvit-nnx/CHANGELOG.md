# Changelog

Changes that reach users of the `canvit-nnx` package, newest first. A release moves the Unreleased
entries under its version. Entries marked **Breaking** change what existing code or saved files do;
**Hub** entries are steps on the Hugging Face Hub that must follow the release, in order, because
released versions load each repo's `main` revision.

## Unreleased

### Added

- `canvit_nnx.specialize.in1k`: ImageNet-1k fine-tuning (LP-FT) of CanViT-B with the paper's
  recipe, on one host or a multi-host slice.
- `canvit_nnx.detect`: COCO 2017 object detection with a CenterNet head on the canvas.
  `CanViTForObjectDetection` loads and saves in the checkpoint format; `canvit_nnx.detect.train`
  trains it end to end and `canvit_nnx.detect.evaluate` scores COCO average precision after each
  glimpse.
- The `train` extra, with the dependencies of the two trainers: optax, pycocotools, tfrecord and
  tyro.
- `canvit_nnx.policies`: the F-IID, R-IID and C2F viewing policies (`full_then_random`,
  `all_random`, `coarse_to_fine`), with `random_viewpoint`, pretraining's random viewpoint down to a
  `min_scale`, and `quadtree_level`, the tiles of a C2F level.
- `canvit_nnx.releases`: the Hub ids of the released NNX checkpoints and their viewing geometry.
- `canvit_nnx.training`, the trainers' shared machinery: data parallelism over one host or a
  multi-host slice, run directories (recipe, metrics, images, checkpoints with resumption), the
  training loop and its hardware FLOPs utilization.
- `compute_dtype` on `CanViT`, `CanViTForImageClassification` and its `from_pretrained_with_probe`:
  `"float32"` (default, outputs unchanged) or `"bfloat16"` with the fine-tuning's mixed-precision
  rules.
- Canvas Attention as fused Pallas kernels in bfloat16 on TPU.
- `canvit_nnx.hub.save_state` and `load_state`: strict safetensors state files for any NNX module or
  optimizer; optimizer states without data, such as optax's masked-out parameter groups, are left
  out.

### Changed

- **Breaking**: Canvas Attention's Write returns the updated canvas, and its Read the updated
  glimpse tokens, instead of the residual updates. Code calling these modules directly adds nothing
  itself any more. Their arguments are now `tokens`, `canvas`, `tokens_rope` and `canvas_rope`
  (formerly `query`, `kv`, `query_rope` and `kv_rope`), and their base class `CanvasAttention` is
  gone.
- **Breaking**: the building blocks `PatchEmbed`, `ViTBlock`, `ViTBackbone`, `CanvasAttentionRead`
  and `CanvasAttentionWrite` take a required `dtype`, the compute dtype that `CanViT` passes them.
- **Breaking**: `CanViT` requires the recurrent state in its compute dtype, as `init_state` makes
  it; a state of another dtype fails instead of being promoted.
- Glimpses are sampled by separable bilinear interpolation: the same function. Float32 outputs
  differ by a few ulps per pixel (outputs of the released models move at that level); bfloat16
  inputs are interpolated with bfloat16 weights instead of in float32.

### Hub

- After the release that includes `canvit_nnx.detect`: create the COCO detector repositories from
  their staged conversions, each with its generated card.
