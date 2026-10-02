# Visualizations and web bundles

The visualization package records model outputs once and writes bundles that
the project page, slides and paper tooling can render. A recorded rollout
contains the scene, viewpoint, crop, canvas, segmentation readout and optional
Canvas Attention Write details for every glimpse. A path bundle records the
same readout at samples along a smooth viewpoint path. The live export is the
browser runtime: an ONNX graph for one CanViT glimpse and a separate ONNX graph
for the segmentation readout.

The maintained consumers are:

```text
canvit_pytorch.viz.record       -> Rollout records
canvit_pytorch.viz.pca          -> PCA colors for token grids
canvit_pytorch.viz.web          -> manifest.json and PNG layers
canvit_pytorch.viz.path         -> closed Bézier viewpoint paths
canvit_pytorch.viz.path_bundle  -> carried/reset path atlases
canvit_pytorch.viz.live         -> ONNX graphs, parity and browser checks
```

`site/record_bundles.sh` records the bundles used by the project page. The
web components under `site/js/canvit/` consume the generated manifests and
layers.

## Run the recorders

From `canvit-pytorch/`, install the optional dependencies and inspect the
commands:

```bash
uv sync --extra viz
uv run --extra viz python -m canvit_pytorch.viz --help
uv run --extra live python -m canvit_pytorch.viz.live --help
```

`rollout` requires an image, source metadata, output path and title. The
annotation is a matching ADE20K label PNG or `None`:

```bash
uv run --extra viz python -m canvit_pytorch.viz rollout \
  --scene.image IMAGE \
  --scene.annotation None \
  --scene.source SOURCE \
  --scene.attribution ATTRIBUTION \
  --scene.license LICENSE \
  --scene.out OUTPUT \
  --scene.title TITLE
```

Use `--policy`, `--num-glimpses`, `--pca-protocol`, `--capture-writes` and
`--seed` to select the rollout. The defaults are the 21-glimpse
entropy-guided coarse-to-fine policy, fixed PCA limits and seed 0. `path`
uses the same scene options and adds `--keypoints`, a sequence of
`row col scale` triples, plus `--num-samples` and `--duration-ms`.

## Shared conventions

- Viewpoints are `(row, col, scale)` in `[-1, 1]`. Rows and columns follow
  tensor indexing. A crop box is stored as `top`, `left` and `size`, each as a
  fraction of the square scene; Python computes this geometry before writing
  the bundle.
- The segmentation readout always uses the layer-normalized canvas patches.
  Canvas Attention operates on its canvas state; the raw canvas is never used
  as a segmentation readout or displayed canvas map.
- A recorded bundle stores model outputs. The browser applies palettes and
  colormaps to the stored layers; it does not run the model.
- Class labels are 8-bit grayscale PNGs. In a web bundle, entropy and canvas
  change are 16-bit grayscale PNGs containing fractions of their display
  ranges. Entropy is predictive entropy divided by `log(num_classes)`. Canvas
  change is one minus the cosine similarity of each layer-normalized canvas
  patch before and after the glimpse, divided by 2 for storage in `[0, 1]`.
- `truth.png` stores the annotation class at the center pixel of each canvas
  cell. Its value is 255 for an ignored or unlabeled cell. A glimpse's
  `pixel_accuracy` is computed over annotated scene pixels after bilinear
  upsampling of the logits to the scene resolution.

## Web bundle

The web-bundle schema is
`canvit-web-bundle-70766501-fb7f-4856-8a88-bb16253c34c5`. A reader must reject
a different schema. The generated directory contains `scene.png`,
`initial_canvas.png`, `manifest.json` and a `tNN/` directory per glimpse.

The manifest records:

- `scene`: the scene image, square pixel size, source, attribution, license
  and optional annotation source and `truth.png`;
- `model`: the checkpoint identity and architecture facts used by the
  renderer;
- `readout`: the segmentation probe identity, class count and class names;
- `policy`: the policy name, paper name, description, determinism and seed;
- `canvas_grid` and `glimpse_px`;
- `pca`: either `paper` or `fixed-limits`, with a basis fitted on the last
  glimpse's canvas;
- `initial_canvas` and `glimpses`, where each glimpse has `t`, `viewpoint`,
  `box`, its layer paths and optional `pixel_accuracy`; and
- `provenance`, including the code revision and runtime that produced it.

Each glimpse always has `crop.png`, `canvas.png`, `labels.png`, `entropy.png`
and `change.png`. With `--capture-writes`, it also has, for each Canvas
Attention Write, `writeK_glimpse.png`, `writeK.png` and `writeK_canvas.png`.
The first is the glimpse patch-token source for that Write, the second is the
Write residual, and the third is the canvas after that Write. The latter uses
the canvas layer's basis and color limits; the final Write canvas equals the
glimpse's `canvas.png`.

`pca.protocol` controls color scaling:

- `paper` fits the basis on the last glimpse and computes min/max limits for
  each frame;
- `fixed-limits` uses the same basis and one set of limits across all glimpses,
  so an animation keeps a stable color scale.

`initial_canvas.png` uses the learned initial canvas patch broadcast over every
cell. The `model` record identifies the checkpoint revision and records the
backbone, patch size, canvas width and Read/Write block indices. The `readout`
record identifies the probe revision, class count and class names. The
`provenance` record identifies the code revision and runtime.

## Path bundle

The path-bundle schema is
`canvit-path-bundle-df96391b-61d4-49cd-a1e5-5a4af9a42e77`. `path.keypoints`
are the requested viewpoints, `path.segments` are the four control points of
each closed cubic Bézier segment, `path.viewpoints` are the sampled
`(row, col, scale)` triples and `path.duration_ms` is the animation duration.
The path code shortens Bézier handles to keep
every control point in the valid crop set, so every sampled crop stays inside
the scene.

The generated directory contains `scene.png`, `inputs.jpg`, optional
`truth.png`, one atlas for each `canvas`, `labels` and `entropy` layer in each
condition, and `manifest.json`. `atlas_columns` gives the row-major tile
layout. The `carried` condition carries the canvas from the previous sample;
the `reset` condition starts every sample from the learned initial state.
Both conditions use the PCA basis and 1st/99th-percentile limits from one full
scene glimpse. `inputs.jpg` is a display copy of the input crops. Path entropy
atlases are 8-bit, with entropy divided by `log(num_classes)` and scaled to
255.

## Live model

`<canvit-live>` runs a released CanViT-B and its ADE20K probe at a 64×64 canvas.
Each interaction runs the CanViT glimpse graph and then the probe graph on the
new canvas. CanViT and the probe are exported, checked and published as
separate repositories. The graphs are float32 and use the exported geometry;
the export command accepts overrides for all three sizes. The released model
and probe repositories are listed in the [Hugging Face
organization](https://huggingface.co/canvit).

```bash
uv run --extra live python -m canvit_pytorch.viz.live export \
  --out-dir ../site/.live-model

uv run --extra live python -m canvit_pytorch.viz.live parity \
  --model-dir ../site/.live-model \
  --image "$ADE20K_ROOT/images/validation/ADE_val_00001780.jpg"
```

Serve `site/` over HTTP so the page and parity directory are fetchable. Then
run the browser check in each supported backend:

```bash
uv run --extra live python -m canvit_pytorch.viz.live check-browser \
  --page-url http://127.0.0.1:8000/ \
  --reference-url http://127.0.0.1:8000/.live-model/parity/ \
  --backend webgpu \
  --out outputs/live/webgpu.json

uv run --extra live python -m canvit_pytorch.viz.live check-browser \
  --page-url http://127.0.0.1:8000/ \
  --reference-url http://127.0.0.1:8000/.live-model/parity/ \
  --backend wasm \
  --out outputs/live/wasm.json
```

`parity` compares PyTorch CPU float32 with ONNX Runtime CPU on the same
viewpoints. It writes `parity/report.json`, `scene.png`, `episode.json` and
little-endian float32 reference files under `parity/reference/`. The reference
episode contains the entropy-guided coarse-to-fine episode followed by random
viewpoints with the canvas carried. The check accepts relative L2 error up to
`1e-3` per compared output and requires at least `0.999` class-argmax
agreement. It also checks the browser's closed-loop policy choices, pointer
geometry and cached reload.

Stage the checked graphs for publication with:

```bash
uv run --extra live python -m canvit_pytorch.viz.live publish \
  --model-dir ../site/.live-model \
  --out-dir staging
```

The command requires a parity report made from the same graph hashes and
current checkpoint revisions. It stages the two repositories under `staging`;
add `--push` only when those staged files are ready to upload.

### Export files and shapes

`export` writes two directories:

- `canvit/` contains `canvit_step.onnx`, `initial_state.bin` and
  `manifest.json`. The step graph accepts `scene [1, 3, S, S]` in ImageNet
  normalization, `canvas [1, R + G², D]`, `recurrent_cls [1, 1, D_b]`,
  `centers [1, 2]` and `scales [1]`. It returns `next_canvas`,
  `next_recurrent_cls` and `glimpse [1, 3, g, g]`.
- `probe/` contains `probe.onnx` and `manifest.json`. The readout accepts the
  CanViT graph's `next_canvas` and returns `logits [1, C, G, G]` and
  `entropy [1, G, G]` in nats. The probe reads the layer-normalized canvas;
  its graph contains the probe readout weights.

`initial_state.bin` is little-endian float32. Its manifest lists the flattened
parts in order: `init_canvas_registers`, `init_canvas_patch` and
`init_recurrent_cls`. The browser broadcasts `init_canvas_patch` over the
canvas grid in the same way as `CanViT.init_state`.

The CanViT manifest also records the ImageNet normalization mean and standard
deviation, the minimum viewpoint scale, the graph input/output shapes and the
file byte counts and SHA-256 digests. The probe manifest records the same file
integrity fields, the readout metadata and the policy levels used by
entropy-guided coarse-to-fine.

Both graphs embed their weights and use ONNX opset 18. The entropy graph
implements log-softmax as `logits - logsumexp` because ONNX Runtime Web 1.30.0's
WebGPU execution provider lacks a LogSoftmax kernel. The exported graph also
relies on that runtime's GridSample support for opsets 16 through 19.

The probe manifest carries the entropy-guided coarse-to-fine policy. Its
`levels` list contains the quadtree tiles as `(row, col, scale)` viewpoints;
the browser policy reads the probe entropy and chooses among those tiles.

The page validates each manifest's file size and SHA-256 before loading it.
The page constants `PUBLISHED_CANVIT` and `PUBLISHED_PROBE` in
`site/js/canvit/live-model.js` select the published exports.
