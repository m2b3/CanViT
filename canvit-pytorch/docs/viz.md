# Visualizations and the web bundle

One pipeline serves the project page, slides and papers:

```
released model + policy + image
  → canvit_pytorch.viz.record   a Rollout: per glimpse, viewpoint, crop, canvas patches,
                                segmentation logits, what each Canvas Attention Write added
  → canvit_pytorch.viz.pca      the paper's PCA protocol (and a fixed-limit variant)
  → canvit_pytorch.viz.web      a web bundle: manifest.json + PNG layers per glimpse
  → site/js/canvit/             <canvit-*> web components, used unchanged by the site and slides

released model + image + keypoint viewpoints
  → canvit_pytorch.viz.path         a closed Bézier curve through them, sampled densely
  → canvit_pytorch.viz.path_bundle  a path bundle: per sample, the canvas carried and reset
  → site/js/canvit/path.js          <canvit-path>

released model + ADE20K probe
  → canvit_pytorch.viz.live         one glimpse as an ONNX graph (fp32), its initial state, manifest.json
  → site/js/canvit/live.js          <canvit-live>: the model itself, run by ONNX Runtime Web
```

`site/record_bundles.sh` records every bundle the project page shows.

```bash
uv run --extra viz python -m canvit_pytorch.viz --help
uv run --extra live python -m canvit_pytorch.viz.live --help
```

## Principles

- **Paper version first.** Policies come from `canvit_pytorch.policies` (the
  paper's, EG-C2F included), labels from the released ADE20K probe, canvas
  colors from the paper's PCA protocol. The one extension, fixed color limits
  across glimpses, is named in `pca.protocol`.
- **Record once, render anywhere.** For bundles the browser never recomputes
  model outputs; it colors and animates recorded ones. A bundle is generated,
  never edited by hand, and names the checkpoint revisions, code commit, torch
  version and device that produced it, and where its image comes from. The
  live model (below) is the one place the browser runs CanViT; its export is
  compared with PyTorch in Python and in the browser.
- **Geometry is computed once, in Python.** Viewpoints are `(row, col, scale)`
  in `[-1, 1]` (`canvit_pytorch.viewpoint`); the bundle also stores each
  glimpse's box as `top, left, size` fractions of the scene, so no renderer
  converts coordinates. `<canvit-live>` turns pointer positions into
  viewpoints itself; the browser check asserts that each input glimpses where
  it aimed.
- **Data layers are lossless and uncolored.** Class labels are 8-bit
  grayscale PNGs; scalar maps are 16-bit grayscale PNGs, fractions of 65535.
  Entropy is divided by `log(num_classes)`; change is one minus the cosine
  similarity of each layer-normalized canvas patch before and after the
  glimpse, divided by 2. Readouts never use the raw canvas. The browser applies palettes and
  colormaps, so themes and legends live in one place.
- **Maps sit beside the photograph**, never over it.

## Web bundle

`manifest.json` names its format in `schema`: `canvit-web-bundle-70766501-fb7f-4856-8a88-bb16253c34c5`.
A change that breaks readers gets a new identifier.

```json
{
  "schema": "canvit-web-bundle-70766501-fb7f-4856-8a88-bb16253c34c5",
  "title": "…",
  "scene": {"image": "scene.png", "px": 512, "source": "…", "attribution": "…", "license": "…",
            "annotation_source": "ADE_val_….png", "truth": "truth.png"},
  "model": {"repo": "canvit/…", "revision": "<Hub commit>", "backbone": "vitb16", "num_blocks": 12,
            "patch_px": 16, "canvas_dim": 1024, "read_after_blocks": [1, 5, 9], "write_after_blocks": [3, 7, 11]},
  "readout": {"kind": "ade20k-segmentation", "repo": "canvit/probe-…", "revision": "…",
              "num_classes": 150, "class_names": ["wall", "…"]},
  "policy": {"name": "entropy_coarse_to_fine", "paper_name": "EG-C2F", "description": "…",
             "deterministic": true, "seed": 0},
  "canvas_grid": 64,
  "glimpse_px": 128,
  "pca": {"protocol": "fixed-limits", "basis": "last-glimpse"},
  "initial_canvas": "initial_canvas.png",
  "glimpses": [
    {"t": 0,
     "viewpoint": {"row": 0.0, "col": 0.0, "scale": 1.0},
     "box": {"top": 0.0, "left": 0.0, "size": 1.0},
     "pixel_accuracy": 0.867,
     "layers": {"crop": "t00/crop.png", "canvas": "t00/canvas.png", "labels": "t00/labels.png",
                "entropy": "t00/entropy.png", "change": "t00/change.png"}}
  ],
  "provenance": {"created": "…", "git_commit": "…", "git_dirty": false, "torch": "…", "device": "cpu",
                 "precision": "float32", "…": "…"}
}
```

Block indices count from 0: a Read after block 1 follows the second block.
When the scene has an ADE20K annotation, `truth.png` holds its class at the
center pixel of each canvas cell (255 where unlabeled), and each glimpse's
`pixel_accuracy` is the fraction of annotated scene pixels labeled correctly,
with the logits upsampled bilinearly to the scene as in the paper's evaluation.
With `--capture-writes`, each glimpse also has `write0`, `write1`, … layers:
a PCA image of what each Canvas Attention Write added to the canvas, one basis
per Write, as in the paper's canvas-evolution figure; and `write0_canvas`,
`write1_canvas`, … layers: the canvas after that Write, in the `canvas` layer's
basis and color limits, the last one equal to the glimpse's `canvas`; and `write0_glimpse`, `write1_glimpse`, …
layers: the glimpse's patch tokens each Write read, one cell per patch, one PCA basis per Write.

`initial_canvas` is the canvas before the first glimpse, every patch the same
learned vector, in the `canvas` layers' basis and the color limits they share
under `"fixed-limits"`.

`pca.protocol` is `"paper"` (basis fitted on the last glimpse's canvas,
min-max per frame, as in the paper's Figure 1) or `"fixed-limits"` (same
basis, one set of color limits for every glimpse, so animations do not
flicker; an extension).

## Path bundle

`schema`: `canvit-path-bundle-df96391b-61d4-49cd-a1e5-5a4af9a42e77`. One glimpse
per sample along a closed cubic Bézier curve through keypoint viewpoints; the
curve's control points stay inside the valid viewpoints, a convex set, so every
sample is a crop inside the scene. Each sample is recorded twice: with the
canvas carried from the previous sample (`carried`) and from the initial canvas
(`reset`, what that glimpse alone gives). Layers are atlases, one tile per
sample in row-major order, `atlas_columns` tiles per row.

```json
{
  "schema": "canvit-path-bundle-df96391b-61d4-49cd-a1e5-5a4af9a42e77",
  "title": "…",
  "scene": {"image": "scene.png", "px": 512, "source": "…", "attribution": "…", "license": "…",
            "annotation_source": "…", "truth": "truth.png"},
  "model": {"…": "as in the web bundle"},
  "readout": {"…": "as in the web bundle"},
  "path": {"keypoints": [[-0.2, -0.65, 0.3], "…"], "segments": [[[-0.2, -0.65, 0.3], "… 4 control points per segment"], "…"],
           "duration_ms": 5000, "viewpoints": [[-0.2, -0.65, 0.3], "… one (row, col, scale) per sample"]},
  "canvas_grid": 64,
  "glimpse_px": 128,
  "atlas_columns": 18,
  "inputs": "inputs.jpg",
  "pca": {"basis": "canvas after one full-scene glimpse from the initial state",
          "limits": "1st and 99th percentiles of that canvas's projection, fixed for all samples"},
  "conditions": {
    "carried": {"layers": {"canvas": "carried-canvas.png", "labels": "carried-labels.png",
                           "entropy": "carried-entropy.png"}, "pixel_accuracy": [0.68, "…"]},
    "reset": {"…": "the same, from the initial canvas at every sample"}
  },
  "provenance": {"…": "as in the web bundle"}
}
```

`inputs.jpg` holds the model's input crops (JPEG, for display only). Entropy
tiles are 8-bit, entropy divided by `log(num_classes)` times 255. A renderer
moves the viewpoint continuously along `segments` (each segment takes an equal
share of `duration_ms`) and changes the maps at the samples.

## Live model

`<canvit-live>` runs the released CanViT-B and its ADE20K probe (64×64
canvas) in the browser: per glimpse, the CanViT graph, then the probe graph on
the new canvas. CanViT and the probe are exported, checked and published
apart. Both graphs are float32; a lossy variant (float16, quantization) is not
used until `parity` and `check-browser` pass on it.

```bash
# from canvit-pytorch/; $ADE20K_ROOT is the ADEChallengeData2016 directory
uv run --extra live python -m canvit_pytorch.viz.live export --out-dir ../site/.live-model
uv run --extra live python -m canvit_pytorch.viz.live parity --model-dir ../site/.live-model \
    --image $ADE20K_ROOT/images/validation/ADE_val_00001780.jpg
# serve site/ (site/README.md); live.html with model=".live-model/canvit" probe=".live-model/probe", then,
# with Google Chrome installed:
uv run --extra live python -m canvit_pytorch.viz.live check-browser --page-url http://127.0.0.1:8000/live.html \
    --reference-url http://127.0.0.1:8000/.live-model/parity/ --backend webgpu --out outputs/live/webgpu.json
# stage, then with --push upload, to hub.repos.LIVE_CANVIT and LIVE_PROBE (new private repos, made public once reviewed)
uv run --extra live python -m canvit_pytorch.viz.live publish --model-dir ../site/.live-model --out-dir staging
```

`export` writes two directories; `site/.live-model/` is ignored by git, and
the exported graphs are never committed.

- `canvit/`: `canvit_step.onnx` (`viz.live.step.GlimpseStep`: inputs `scene`
  [1, 3, S, S] (ImageNet-normalized), `canvas`, `recurrent_cls`, `centers`
  [1, 2] (row, col), `scales` [1]; outputs `next_canvas`,
  `next_recurrent_cls` and `glimpse` [1, 3, g, g], cropped from the scene
  inside the graph by `sample_at_viewpoint`); `initial_state.bin`,
  little-endian float32, the parts listed in the manifest in order (CanViT's
  `init_canvas_registers`, `init_canvas_patch`, `init_recurrent_cls`), from
  which the page builds the initial canvas as `CanViT.init_state` does; and
  `manifest.json`.
- `probe/`: `probe.onnx` (`viz.live.step.ProbeReadout`: input `canvas`;
  outputs `logits` [1, C, G, G], the probe on the layer-normalized canvas,
  and `entropy` [1, G, G] in nats) and `manifest.json`. Only the probe's
  weights are in its graph.

Both graphs embed their weights and use opset 18, and entropy is
`predictive_entropy` with `log_softmax` written as `logits - logsumexp`:
ONNX Runtime Web 1.30.0's WebGPU execution provider has no LogSoftmax kernel
and implements GridSample for opsets 16 to 19.

`parity` runs PyTorch (CPU) on one image, an EG-C2F episode followed by
glimpses at random viewpoints with the canvas carried, replays the viewpoints
through ONNX Runtime (CPU), both graphs in turn, and fails when an output's
relative L2 difference exceeds `parity.MAX_REL_L2` or fewer than
`parity.MIN_ARGMAX_AGREEMENT` of the cells keep their class. It writes
`parity/` beside the two directories: a report (with both graphs' SHA-256),
the scene as the reference saw it, the viewpoints and the reference outputs.
`check-browser` drives `<canvit-live>` in headless Chrome as a visitor would
(download button, clicks after scrolling, a drag, the keyboard), replays the
reference viewpoints, lets the element's EG-C2F choose and compares its
choices with PyTorch's, and times whole EG-C2F episodes; it fails on the same
thresholds as `parity`, on a missed EG-C2F choice or a misplaced glimpse.
`--backend wasm` starts Chrome without a GPU process, so the page falls back
to WebAssembly as it does for visitors without WebGPU. The page it checks may
run the local export or the published ones.

`publish` refuses an export without a passing parity report on these very
graphs, or whose model or probe is no longer the Hub's current revision. It
stages each repo with a card from its manifest and the parity report
(`hub.cards.live_canvit_card`, `live_probe_card`). The project page and the
talks run the published exports (`PUBLISHED_CANVIT`, `PUBLISHED_PROBE` in
`site/js/canvit/live-model.js`).

### Manifests

Each directory's `manifest.json` says what the page needs; `export` writes
them, and `viz.live.export.read_manifests` and `site/js/canvit/live-model.js`
check their schemas and that the probe reads the canvas CanViT writes.

```json
{
  "schema": "canvit-live-canvit-0a49b16e-2f86-41ae-8b08-f83d94fb1732",
  "graph": {"path": "canvit_step.onnx", "bytes": "…", "sha256": "…", "opset": 18,
            "inputs": {"scene": [1, 3, 512, 512], "…": "…"}, "outputs": {"next_canvas": [1, 4112, 1024], "…": "…"}},
  "initial_state": {"path": "initial_state.bin", "bytes": "…", "sha256": "…", "dtype": "float32",
                    "parts": [{"name": "init_canvas_registers", "shape": [16, 1024]}, "…"]},
  "scene_px": 512, "glimpse_px": 128, "canvas_grid": 64, "num_canvas_registers": 16,
  "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
  "min_scale": 0.05,
  "model": {"…": "as in the web bundle"},
  "provenance": {"…": "as in the web bundle", "onnx": "…", "onnxscript": "…"}
}
```

```json
{
  "schema": "canvit-live-probe-23117c20-1e8e-4edb-a56d-21018d157b8d",
  "graph": {"path": "probe.onnx", "…": "…", "inputs": {"canvas": [1, 4112, 1024]}, "outputs": {"logits": [1, 150, 64, 64], "…": "…"}},
  "canvas_grid": 64, "num_canvas_registers": 16,
  "readout": {"…": "as in the web bundle"},
  "policy": {"name": "entropy_coarse_to_fine", "paper_name": "EG-C2F", "description": "…", "num_glimpses": 21,
             "levels": [[[0.0, 0.0, 1.0]], [[-0.5, -0.5, 0.5], "… 4 tiles"], ["… 16 tiles"]]},
  "provenance": {"…": "…"}
}
```

The page checks each file's size and SHA-256 against its manifest before
using it. `min_scale` is pretraining's smallest viewpoint scale
(`policies.random.MIN_SCALE`), the smallest glimpse the page offers.
`policy.levels` are EG-C2F's quadtree tiles `(row, col, scale)` from
`policies.EntropyGuidedC2F`; the policy comes with the probe because it reads
the probe's entropy, and `site/js/canvit/entropy-guided-c2f.js` chooses among
the tiles as the Python policy does.
